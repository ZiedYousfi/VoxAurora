use regex::Regex;
use serde::Deserialize;
use std::error::Error;
use std::process::{Child, Command};
use std::thread;
use std::time::Duration;
use unicode_normalization::UnicodeNormalization;
use ureq;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use crate::dawg_loader;

pub static DAWGS: Lazy<(
    HashMap<&'static str, daachorse::DoubleArrayAhoCorasick<u32>>,
    HashMap<&'static str, Vec<String>>,
)> = Lazy::new(|| dawg_loader::load_dawgs());

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
    let corrected = merge_separated_words_dawg_regex(&lang_tooled, 2);
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

/// Point d'entrée principal : fusion des mots séparés s'ils apparaissent dans les DAWG.
pub fn merge_separated_words_dawg_regex(text: &str, max_merge: usize) -> String {
    let token_matches = get_token_matches(text);
    println!(
        "Starting merge with tokens: {:?}",
        token_matches.iter().map(|m| m.as_str()).collect::<Vec<_>>()
    );

    let mut result = String::new();
    let mut last_end = 0;
    let mut i = 0;

    while i < token_matches.len() {
        // On essaie de fusionner plusieurs tokens si possible
        if let Some((merged_word, merged_count)) =
            try_merge_tokens(text, &token_matches, i, max_merge)
        {
            // Si fusion possible :
            let token_start = token_matches[i].start();
            result.push_str(&text[last_end..token_start]);
            result.push_str(&merged_word);

            last_end = token_matches[i + merged_count - 1].end();
            i += merged_count;
        } else {
            // Sinon, on recopie simplement le token
            let token_start = token_matches[i].start();
            let token_end = token_matches[i].end();

            result.push_str(&text[last_end..token_start]);
            result.push_str(&text[token_start..token_end]);

            last_end = token_end;
            i += 1;
        }
    }

    // On ajoute la fin du texte (après le dernier token)
    result.push_str(&text[last_end..]);
    result
}

/// Extrait les tokens à partir du texte (lettres + apostrophes).
fn get_token_matches(text: &str) -> Vec<regex::Match<'_>> {
    // Vous pouvez compiler la regex de façon statique si besoin, mais on la recrée ici pour la démo
    let re = Regex::new(r"\p{L}+(?:[’']\p{L}+)*").unwrap();
    re.find_iter(text).collect()
}

/// Tente une fusion de plusieurs tokens successifs, en commençant par la taille max_merge,
/// puis en diminuant. Retourne Some((mot_fusionné, nombre_de_tokens_fusionnés)) ou None.
fn try_merge_tokens(
    text: &str,
    token_matches: &[regex::Match<'_>],
    start_index: usize,
    max_merge: usize,
) -> Option<(String, usize)> {
    // On parcourt de la taille max jusqu'à 2 (fusion d'au moins 2 tokens)
    for merge_len in (2..=max_merge).rev() {
        if start_index + merge_len <= token_matches.len() {
            // Vérifie que les tokens sont consécutifs (séparés seulement par des espaces/blancs)
            if !are_tokens_adjacent(text, token_matches, start_index, merge_len) {
                continue;
            }

            // Construit la version fusionnée et la version “espacée” (pour comparer)
            let (candidate, candidate_with_space, candidate_lower, candidate_with_space_lower) =
                build_candidates(text, token_matches, start_index, merge_len);

            println!(
                "Checking candidate: '{}' (from '{}')",
                candidate_lower, candidate_with_space_lower
            );

            // Filtre simple : on abandonne si le mot n'est pas 'raisonnable'
            if !is_reasonable_word(&candidate_lower) {
                println!("⛔ Ignored candidate: '{}' (unreasonable)", candidate_lower);
                continue;
            }

            // Vérifie si le candidat existe dans au moins un DAWG
            let (in_dawg, spaced_in_dawg) =
                check_in_dawg(&candidate_lower, &candidate_with_space_lower);

            if in_dawg {
                let merge_score = compute_merge_score(&candidate_lower, merge_len);
                println!(
                    "🔎 Merge score for '{}': {:.2}",
                    candidate_lower, merge_score
                );

                // Vérifie la logique de fusion (score, version espacée, etc.)
                if let Some((word, count)) = handle_merge_decision(
                    &candidate,
                    &candidate_lower,
                    &candidate_with_space,
                    spaced_in_dawg,
                    merge_len,
                    merge_score,
                ) {
                    return Some((word, count));
                }
            } else {
                println!("⛔ Not merging: word not in dictionary");
            }
        }
    }
    None
}

/// Vérifie que les tokens (sur la plage [start_index..start_index+merge_len]) sont contigus
/// et uniquement séparés par des espaces/blancs.
fn are_tokens_adjacent(
    text: &str,
    token_matches: &[regex::Match<'_>],
    start_index: usize,
    merge_len: usize,
) -> bool {
    for j in start_index..(start_index + merge_len - 1) {
        let gap = &text[token_matches[j].end()..token_matches[j + 1].start()];
        if !gap.chars().all(|c| c.is_whitespace()) {
            return false;
        }
    }
    true
}

/// Construit les différentes variantes (fusionnée, espacée, minuscule, etc.)
fn build_candidates(
    text: &str,
    token_matches: &[regex::Match<'_>],
    start_index: usize,
    merge_len: usize,
) -> (String, String, String, String) {
    let tokens_to_merge: Vec<_> = token_matches[start_index..start_index + merge_len]
        .iter()
        .map(|m| m.as_str())
        .collect();

    let candidate = tokens_to_merge.join(""); // ex: "jourd'hui"
    let candidate_with_space = tokens_to_merge.join(" "); // ex: "jour d hui"

    // Normalise + minuscule
    let candidate_lower = candidate.nfkc().collect::<String>().to_lowercase();
    let candidate_with_space_lower = candidate_with_space
        .nfkc()
        .collect::<String>()
        .to_lowercase();

    (
        candidate,
        candidate_with_space,
        candidate_lower,
        candidate_with_space_lower,
    )
}

/// Vérifie la présence du mot (et de sa version espacée) dans au moins un DAWG.
fn check_in_dawg(candidate_lower: &str, candidate_with_space_lower: &str) -> (bool, bool) {
    let mut in_dawg = false;
    let mut spaced_in_dawg = false;

    for (lang, dawg) in DAWGS.0.iter() {
        if dawg_loader::contains_exact(dawg, candidate_lower) {
            println!("Found '{}' in {} DAWG", candidate_lower, lang);
            in_dawg = true;
        }
        if dawg_loader::contains_exact(dawg, candidate_with_space_lower) {
            println!(
                "Found spaced version '{}' in {} DAWG",
                candidate_with_space_lower, lang
            );
            spaced_in_dawg = true;
        }

        if let Some(word_list) = DAWGS.1.get(lang) {
            if dawg_loader::is_most_similar(word_list, candidate_lower, 1) {
                println!("Found '{}' similar in {} DAWG", candidate_lower, lang);
                in_dawg = true;
            }
        }
    }

    (in_dawg, spaced_in_dawg)
}

/// Décide si on doit fusionner les tokens, selon différentes conditions (score, version espacée, etc.).
/// Retourne Some((candidate, merge_len)) si on fusionne, sinon None.
fn handle_merge_decision(
    candidate: &str,
    candidate_lower: &str,
    candidate_with_space: &str,
    spaced_in_dawg: bool,
    merge_len: usize,
    merge_score: f32,
) -> Option<(String, usize)> {
    // Cas particulier: fusion de 2 mots courts (< 10 lettres)
    let short_common_word = (merge_len == 2) && (candidate_lower.len() < 10);

    if short_common_word {
        let bert_score = check_word_with_bert(candidate_lower).unwrap_or(0.0);
        if spaced_in_dawg && bert_score < 0.1 {
            println!(
                "⛔ Not merging common expression: '{}' (keeping '{}') [BERT score: {:.2}]",
                candidate, candidate_with_space, bert_score
            );
            None
        } else {
            println!(
                "✨ Merging short word: '{}' [score: {:.2}]",
                candidate, merge_score
            );
            Some((candidate.to_string(), merge_len))
        }
    } else {
        let threshold = match merge_len {
            2 => 0.70,
            3 => 0.75,
            _ => 0.80,
        };

        if !spaced_in_dawg || merge_score >= threshold {
            println!(
                "✨ Merging: '{}' [score: {:.2} ≥ {:.2}]",
                candidate, merge_score, threshold
            );
            Some((candidate.to_string(), merge_len))
        } else {
            println!(
                "⛔ Not merging: spaced version exists and score {:.2} < {:.2}",
                merge_score, threshold
            );
            None
        }
    }
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
        2 => 0.50,
        3 => 0.55,
        _ => 0.60,
    };

    // Pénalité légère pour mot très court
    let length_penalty = if len < 5 { -0.05 } else { 0.0 };

    // Score BERT (approx. 0..1)
    let bert_score = match check_word_with_bert(word) {
        Ok(s) => s * 0.10, // on pondère le score BERT pour éviter l'effet “tout ou rien”
        Err(_) => {
            println!("⚠️ BERT check failed for '{}'", word);
            0.0
        }
    };

    let total = base_score + length_penalty + bert_score;
    total.clamp(0.0, 1.0)
}

/// Vérifie la plausibilité d'un mot via BERT en calculant la norme de l'embedding
/// sur plusieurs contextes. Retourne un score [0..1].
fn check_word_with_bert(word: &str) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
    // Mot de référence courant et valide
    const REFERENCE_WORD: &str = "bonjour";

    // Plusieurs contextes différents pour mieux évaluer le mot
    let contexts = [
        format!("On utilise souvent le mot {}.", "{}"), // Fin de phrase
        format!("Le {} est un terme courant en français.", "{}"), // Début de phrase, sujet
        format!("J'aime beaucoup ce {}.", "{}"),        // Complément d'objet
        format!("Il parle de {} avec enthousiasme.", "{}"), // Après préposition
    ];

    let mut total_score = 0.0;

    // Pour chaque contexte, calculer un score
    for context_template in &contexts {
        let reference_context = context_template.replace("{}", REFERENCE_WORD);
        let test_context = context_template.replace("{}", word);

        // Obtenir les embeddings
        let reference_embedding = super::bert::encode_sentence(&reference_context)?;
        let test_embedding = super::bert::encode_sentence(&test_context)?;

        // Calcul des normes L2
        let reference_norm = reference_embedding
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        let test_norm = test_embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();

        // Calcul de similarité cosinus
        let dot_product: f32 = reference_embedding
            .iter()
            .zip(test_embedding.iter())
            .map(|(&a, &b)| a * b)
            .sum();

        let cosine_similarity = dot_product / (reference_norm * test_norm).max(1e-6);

        // Ratio des normes (pour détecter les anomalies)
        let norm_ratio = (test_norm / reference_norm).clamp(0.0, 2.0) / 2.0;

        // Score pour ce contexte
        let context_score = (cosine_similarity * 0.7 + norm_ratio * 0.3).clamp(0.0, 1.0);
        total_score += context_score;

        println!("  - Contexte '{}': norm = {:.2}", test_context, test_norm);
        println!(
            "  - Référence '{}': norm = {:.2}",
            reference_context, reference_norm
        );
        println!("  - Similarité cosinus: {:.2}", cosine_similarity);
        println!("  - Ratio des normes: {:.2}", norm_ratio);
        println!("  - Score pour ce contexte: {:.2}", context_score);
    }

    // Score final: moyenne des scores pour chaque contexte
    let combined_score = (total_score / contexts.len() as f32).clamp(0.0, 1.0);
    println!("  => BERT score combiné final = {:.2}", combined_score);

    Ok(combined_score)
}
