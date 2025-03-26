use serde::Deserialize;
use std::error::Error;
use std::process::{Child, Command};
use std::thread;
use std::time::Duration;
use ureq;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

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
    let corrected = burt_correct_text(clean.trim());
    println!("Texte après correction : {}", corrected);
    corrected
}

#[derive(Debug, Deserialize)]
struct Match {
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
            let start = corrected.char_indices().nth(m.offset)
                .map(|(byte_idx, _)| byte_idx)
                .unwrap_or(0);
            let end = corrected.char_indices().nth(m.offset + m.length)
                .map(|(byte_idx, _)| byte_idx)
                .unwrap_or_else(|| corrected.len());
            corrected.replace_range(start..end, &replacement.value);
        }
    }

    corrected
}
