use regex::Regex;
use serde::Deserialize;
use std::error::Error;
use std::process::{Child, Command};
use std::thread;
use std::time::Duration;
use ureq;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::dawg_loader;

/// Lance le serveur LanguageTool en arrière-plan et attend que ce dernier soit opérationnel
pub fn start_languagetool_server() -> Child {
    let child = Command::new("java")
        .args([
            "-cp",
            "tools/LanguageTool-6.6-SNAPSHOT/languagetool-server.jar",
            "org.languagetool.server.HTTPServer",
            "--port",
            "8081",
        ])
        .spawn()
        .expect("Échec du démarrage de LanguageTool");

    // Attendre que le serveur soit opérationnel
    wait_for_languagetool_server().expect("Le serveur LanguageTool n'est pas démarré");
    child
}

/// Vérifie que le serveur LanguageTool répond sur l'endpoint /v2/check
fn wait_for_languagetool_server() -> Result<(), Box<dyn Error>> {
    let base_url = "http://localhost:8081/v2/check";
    let mut attempts = 0;
    while attempts < 10 {
        let request_url = format!(
            "{}?language={}&text={}",
            base_url,
            "fr-FR",
            urlencoding::encode("Bonjour")
        );
        let response_result = ureq::get(&request_url)
            .header("Accept", "application/json")
            .call();
        match response_result {
            Ok(response) if response.status() == 200 => return Ok(()),
            Ok(response) => {
                println!(
                    "Le serveur LanguageTool n'est pas encore opérationnel, tentative {}...",
                    attempts + 1
                );
                println!("reponse: {:?}", response);
                println!("status: {:?}", response.status());
            }
            Err(err) => {
                println!("Erreur lors de l'envoi de la requête: {:?}", err);
            }
        }
        attempts += 1;
        thread::sleep(Duration::from_secs(1));
    }
    Err("LanguageTool server did not start in time".into())
}

pub fn init_model(path_to_model: String) -> Result<WhisperContext, Box<dyn Error>> {
    let ctx = WhisperContext::new_with_params(&path_to_model, WhisperContextParameters::default())?;

    Ok(ctx)
}

pub async fn transcribe(
    model: &WhisperContext,
    audio: &[f32],
    lang: &str,
) -> Result<String, Box<dyn Error>> {
    let mut params = FullParams::new(SamplingStrategy::default());
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_token_timestamps(false);
    params.set_language(Some(lang));

    // Crée un nouvel état pour cette inférence
    let mut state = model.create_state()?;

    // Traite les données audio
    state.full(params, audio)?;

    // Concatène les segments
    let num_segments = state.full_n_segments()?;
    let mut result = String::new();
    for seg in 0..num_segments {
        let num_tokens = state.full_n_tokens(seg)?;
        let mut segment_text = String::new();
        for token in 0..num_tokens {
            let token_text = state.full_get_token_text(seg, token)?;
            let token_text = token_text.trim();
            if !segment_text.is_empty()
                && !token_text.starts_with(|c: char| c.is_ascii_punctuation())
                && !token_text.starts_with("[")
            {
                segment_text.push(' ');
            }
            segment_text.push_str(token_text);
        }
        result.push_str(segment_text.trim());
        result.push(' ');
    }

    // Appel de la fonction de nettoyage
    let cleaned_result = clean_whisper_text(&result);

    Ok(cleaned_result)
}

pub fn clean_whisper_text(original: &str) -> String {
    use regex::Regex;

    // Supprimer les balises spéciales du type [_BEG_] ou [_TT_...]
    let re_beg = Regex::new(r"\[_BEG_\]").unwrap();
    let re_tt = Regex::new(r"\[_TT_\d+\]").unwrap();
    let mut clean = re_beg.replace_all(original, "").to_string();
    clean = re_tt.replace_all(&clean, "").to_string();

    // Retirer les espaces multiples
    let re_spaces = Regex::new(r"\s+").unwrap();
    clean = re_spaces.replace_all(&clean, " ").to_string();

    println!("Texte avant correction : {}", clean);
    // Appel à l'API LanguageTool
    let lang_tooled = burt_correct_text(clean.trim());
    let corrected = merge_separated_words_dawg_regex(&lang_tooled, 3);
    println!("Texte après correction : {}", corrected);
    corrected
}

#[derive(Debug, Deserialize)]
struct Match {
    #[allow(dead_code)]
    message: String,
    replacements: Vec<Replacement>,
    offset: usize,
    length: usize,
}

#[derive(Debug, Deserialize)]
struct Replacement {
    value: String,
}

#[derive(Debug, Deserialize)]
struct LTResponse {
    matches: Vec<Match>,
}

pub fn burt_correct_text(text: &str) -> String {
    let base_url = "http://localhost:8081/v2/check";
    let request_url = format!(
        "{}?language={}&text={}",
        base_url,
        "fr",
        urlencoding::encode(text)
    );
    let body: String = ureq::get(&request_url)
        .header("Accept", "application/json")
        .call()
        .unwrap()
        .body_mut()
        .read_to_string()
        .unwrap();
    let lt_response: LTResponse = serde_json::from_str(&body).unwrap();

    let mut corrected = text.to_string();
    // Sort corrections descending by offset to apply without affecting subsequent indices.
    let mut matches = lt_response.matches;
    matches.sort_by(|a, b| b.offset.cmp(&a.offset));
    for m in matches {
        if let Some(replacement) = m.replacements.first() {
            // Convert character offset and length to byte indices.
            let start = corrected
                .char_indices()
                .nth(m.offset)
                .map(|(byte_idx, _)| byte_idx)
                .unwrap_or(0);
            let end = corrected
                .char_indices()
                .nth(m.offset + m.length)
                .map(|(byte_idx, _)| byte_idx)
                .unwrap_or_else(|| corrected.len());
            corrected.replace_range(start..end, &replacement.value);
        }
    }

    corrected
}

/// Exemple d’accès global aux DAWGs dans un module `super::DAWGS`.
/// Adaptez si besoin : ici on suppose un `Vec<(&'static str, DawgType)>`.
/// let DAWGS: Vec<(&str, Dawg)> = vec![("fr", dawg_fr), ("en", dawg_en)];
/// ...
/// Dans le code ci-dessous, on accède à `super::DAWGS`.
fn is_reasonable_word(word: &str) -> bool {
    word.len() <= 20 && word.chars().all(|c| c.is_alphabetic() || c == '\'')
}

/// Fusionne des tokens contigus du texte original s'ils forment un mot entier présent dans les DAWG.
/// Utilise un score pour décider s'il faut fusionner ou conserver la version espacée.
pub fn merge_separated_words_dawg_regex(text: &str, max_merge: usize) -> String {
    // On détecte les “mots” à l'aide d'une regex (lettres + apostrophes éventuelles).
    let re = Regex::new(r"\p{L}+(?:[’']\p{L}+)*").unwrap();
    let token_matches: Vec<_> = re.find_iter(text).collect();

    let mut result = String::new();
    let mut last_end = 0;
    let mut i = 0;

    println!(
        "Starting merge with tokens: {:?}",
        token_matches.iter().map(|m| m.as_str()).collect::<Vec<_>>()
    );

    while i < token_matches.len() {
        let mut merged = None;

        // On tente d’abord la fusion du plus grand nombre de tokens possible (max_merge), puis on diminue.
        for merge_len in (2..=max_merge).rev() {
            if i + merge_len <= token_matches.len() {
                // Vérif : les tokens sont-ils uniquement séparés par des espaces/blancs ?
                let mut adjacent = true;
                for j in i..(i + merge_len - 1) {
                    let gap = &text[token_matches[j].end()..token_matches[j + 1].start()];
                    if !gap.chars().all(|c| c.is_whitespace()) {
                        adjacent = false;
                        break;
                    }
                }
                if !adjacent {
                    continue;
                }

                // Construit la version fusionnée (sans espace) et la version “espacée”
                let tokens_to_merge: Vec<_> = token_matches[i..i + merge_len]
                    .iter()
                    .map(|m| m.as_str())
                    .collect();
                let candidate = tokens_to_merge.join(""); // "jourd'hui"
                let candidate_with_space = tokens_to_merge.join(" "); // "jour d hui"

                // Pour le matching, on travaille en minuscule (selon le DAWG).
                use unicode_normalization::UnicodeNormalization;
                let candidate_lower = candidate.nfkc().collect::<String>().to_lowercase();
                let candidate_with_space_lower = candidate_with_space
                    .nfkc()
                    .collect::<String>()
                    .to_lowercase();

                println!(
                    "Checking candidate: '{}' (from '{}')",
                    candidate_lower, candidate_with_space_lower
                );

                // Filtre de base : taille, caractères
                if !is_reasonable_word(&candidate_lower) {
                    println!("⛔ Ignored candidate: '{}' (unreasonable)", candidate_lower);
                    continue;
                }

                // On vérifie si c’est un match exact dans au moins un DAWG
                let mut in_dawg = false;
                let mut spaced_in_dawg = false;

                for (lang, dawg) in super::DAWGS.0.iter() {
                    if dawg_loader::contains_exact(dawg, &candidate_lower) {
                        println!("Found '{}' in {} DAWG", candidate_lower, lang);
                        in_dawg = true;
                    }
                    if dawg_loader::contains_exact(dawg, &candidate_with_space_lower) {
                        println!(
                            "Found spaced version '{}' in {} DAWG",
                            candidate_with_space_lower, lang
                        );
                        spaced_in_dawg = true;
                    }

                    // Use word_lists for Levenshtein check instead of DAWG
                    if let Some(word_list) = super::DAWGS.1.get(lang) {
                        if dawg_loader::is_most_similar(word_list, &candidate_lower, 2) {
                            println!("Found '{}' similar in {} DAWG", candidate_lower, lang);
                            in_dawg = true;
                        }
                    }
                }

                if in_dawg {
                    // Calcule le score global
                    let merge_score = compute_merge_score(&candidate_lower, merge_len);
                    println!(
                        "🔎 Merge score for '{}': {:.2}",
                        candidate_lower, merge_score
                    );

                    // Cas particulier : fusion de 2 mots courts (< 10 lettres)
                    let short_common_word = (merge_len == 2) && (candidate_lower.len() < 10);

                    if short_common_word {
                        // On regarde le score BERT pour savoir si on fusionne malgré la version espacée.
                        let bert_score = check_word_with_bert(&candidate_lower).unwrap_or(0.0);
                        if spaced_in_dawg && bert_score < 0.1 {
                            println!(
                                "⛔ Not merging common expression: '{}' (keeping '{}') [BERT score: {:.2}]",
                                candidate, candidate_with_space, bert_score
                            );
                        } else {
                            println!(
                                "✨ Merging short word: '{}' (from {}) [score: {:.2}]",
                                candidate,
                                tokens_to_merge.join(" + "),
                                merge_score
                            );
                            merged = Some((candidate, merge_len));
                            break;
                        }
                    } else {
                        // Seuil selon la taille de la fusion
                        let threshold = match merge_len {
                            2 => 0.70,
                            3 => 0.75,
                            _ => 0.80,
                        };

                        if !spaced_in_dawg || merge_score >= threshold {
                            println!(
                                "✨ Merging: '{}' (from {}) [score: {:.2} ≥ {:.2}]",
                                candidate,
                                tokens_to_merge.join(" + "),
                                merge_score,
                                threshold
                            );
                            merged = Some((candidate, merge_len));
                            break;
                        } else {
                            println!(
                                "⛔ Not merging: spaced version exists and score {:.2} < {:.2}",
                                merge_score, threshold
                            );
                        }
                    }
                } else {
                    println!("⛔ Not merging: word not in dictionary");
                }
            }
        }

        // Construit la chaîne de sortie
        let token_start = token_matches[i].start();
        result.push_str(&text[last_end..token_start]);

        if let Some((word, count)) = merged {
            // On a décidé de fusionner
            result.push_str(&word);
            last_end = token_matches[i + count - 1].end();
            i += count;
        } else {
            // On laisse le token tel quel
            let token_end = token_matches[i].end();
            result.push_str(&text[token_start..token_end]);
            last_end = token_end;
            i += 1;
        }
    }

    // Puis on ajoute la fin du texte après le dernier token
    result.push_str(&text[last_end..]);
    result
}

/// Calcule un score de fusion [0..1].
fn compute_merge_score(word: &str, merge_len: usize) -> f32 {
    let len = word.len();

    // Si la taille est hors [3..20], on renvoie 0
    if !(3..=20).contains(&len) {
        return 0.0;
    }

    // Score de base
    let base_score = match merge_len {
        2 => 0.70,
        3 => 0.75,
        _ => 0.80,
    };

    // Pénalité légère pour mot très court
    let length_penalty = if len < 5 { -0.05 } else { 0.0 };

    // Score BERT (approx. 0..1)
    let bert_score = match check_word_with_bert(word) {
        Ok(s) => s * 0.40, // on pondère le score BERT pour éviter l'effet “tout ou rien”
        Err(_) => {
            println!("⚠️ BERT check failed for '{}'", word);
            0.0
        }
    };

    let total = base_score + length_penalty + bert_score;
    total.clamp(0.0, 1.0)
}

/// Vérifie la plausibilité d'un mot via BERT en calculant la norme de l'embedding
/// sur 2 contextes. Retourne un score [0..1].
fn check_word_with_bert(word: &str) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
    // Deux phrases contextuelles simples
    let contexts = vec![
        format!("On utilise souvent le mot {}.", word),
        format!("Le mot {} apparaît dans les textes officiels.", word),
    ];

    let mut total_norm = 0.0;
    let mut count = 0;

    for ctx in contexts {
        let embedding = super::bert::encode_sentence(&ctx)?;
        // Calcule la norme L2
        let norm = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        total_norm += norm;
        count += 1;

        println!("  - Contexte '{}': norm = {:.2}", ctx, norm);
    }

    if count == 0 {
        return Ok(0.0);
    }

    let mean_norm = total_norm / count as f32;
    // Ex. si la norme BERT typique tourne autour de 10-15, on normalise en divisant par 15
    let scaled_score = (mean_norm / 15.0).clamp(0.0, 1.0);

    println!("  => BERT final (moyenne des normes) = {:.2}", mean_norm);
    Ok(scaled_score)
}
