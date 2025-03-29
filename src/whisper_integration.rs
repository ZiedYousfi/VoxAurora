use crate::dawg_loader;
use crate::bert;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::process::{Child, Command};
use std::thread;
use std::time::Duration;
use unicode_normalization::UnicodeNormalization;
use ureq;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Global DAWGS: a tuple of (AhoCorasick for each language, word lists).
pub static DAWGS: Lazy<(
    HashMap<&'static str, daachorse::DoubleArrayAhoCorasick<u32>>,
    HashMap<&'static str, Vec<String>>,
)> = Lazy::new(|| dawg_loader::load_dawgs());

/// Starts the LanguageTool server in the background and waits until it's ready.
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
        .expect("Failed to start LanguageTool");

    // Wait until the server is responding
    wait_for_languagetool_server().expect("LanguageTool server is not running yet");
    child
}

/// Checks that the LanguageTool server is listening on /v2/check
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
            Ok(response) if response.status() == 200 => {
                log::info!("LanguageTool server responded with 200. Ready to proceed.");
                return Ok(())
            }
            Ok(response) => {
                log::warn!(
                    "LanguageTool server is not ready yet, attempt #{} ...",
                    attempts + 1
                );
                log::warn!("Response: {:?}", response);
                log::warn!("Status code: {:?}", response.status());
            }
            Err(err) => {
                log::warn!("Error while checking LanguageTool server readiness: {:?}", err);
            }
        }

        attempts += 1;
        thread::sleep(Duration::from_secs(1));
    }

    Err("LanguageTool server did not start in time".into())
}

/// Initializes the Whisper model with default parameters.
pub fn init_model(path_to_model: String) -> Result<WhisperContext, Box<dyn Error>> {
    let ctx = WhisperContext::new_with_params(&path_to_model, WhisperContextParameters::default())?;
    Ok(ctx)
}

/// Transcribes an audio segment asynchronously using Whisper.
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

    // Create a new state for this inference
    let mut state = model.create_state()?;

    // Process the audio data
    state.full(params, audio)?;

    // Concatenate all segments
    let num_segments = state.full_n_segments()?;
    let mut result = String::new();
    for seg in 0..num_segments {
        let num_tokens = state.full_n_tokens(seg)?;
        let mut segment_text = String::new();
        for token in 0..num_tokens {
            let token_text = state.full_get_token_text(seg, token)?;
            let token_text = token_text.trim();
            // Add a space if needed, except for punctuation or special markers
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

    // Call our cleaning function
    let cleaned_result = clean_whisper_text(&result);
    Ok(cleaned_result)
}

/// Cleans up Whisper text by removing special tags, normalizing whitespace,
/// and calling LanguageTool for correction.
pub fn clean_whisper_text(original: &str) -> String {
    // Remove special tags like [_BEG_] or [_TT_...]
    let re_beg = Regex::new(r"\[_BEG_\]").unwrap();
    let re_tt = Regex::new(r"\[_TT_\d+\]").unwrap();
    let mut clean = re_beg.replace_all(original, "").to_string();
    clean = re_tt.replace_all(&clean, "").to_string();

    // Remove multiple spaces
    let re_spaces = Regex::new(r"\s+").unwrap();
    clean = re_spaces.replace_all(&clean, " ").to_string();

    log::info!("Text before correction: {}", clean);

    // Call LanguageTool
    let lang_tooled = burt_correct_text(clean.trim());

    // Then merge separated words using DAWG
    let corrected = merge_separated_words_dawg_regex(&lang_tooled, 2);
    log::info!("Text after correction: {}", corrected);

    corrected
}

/// Data structure for the LanguageTool JSON response
#[derive(Debug, Deserialize)]
struct Match {
    #[allow(dead_code)]
    message: String,
    replacements: Vec<Replacement>,
    offset: usize,
    length: usize,
}

/// A single possible replacement
#[derive(Debug, Deserialize)]
struct Replacement {
    value: String,
}

/// A top-level response structure
#[derive(Debug, Deserialize)]
struct LTResponse {
    matches: Vec<Match>,
}

/// Calls LanguageTool to correct the text using the server at port 8081.
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
    let mut matches = lt_response.matches;

    // Sort matches descending by offset so that we replace from the end
    matches.sort_by(|a, b| b.offset.cmp(&a.offset));

    for m in matches {
        if let Some(replacement) = m.replacements.first() {
            // Convert character offset and length to byte indices
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

/// Checks whether a word is "reasonable": length <= 20, only alphabetic or apostrophes
fn is_reasonable_word(word: &str) -> bool {
    word.len() <= 20 && word.chars().all(|c| c.is_alphabetic() || c == '\'')
}

/// Main entry point for merging separated tokens if they appear in the DAWG
pub fn merge_separated_words_dawg_regex(text: &str, max_merge: usize) -> String {
    let token_matches = get_token_matches(text);

    log::info!(
        "Starting merge with tokens: {:?}",
        token_matches.iter().map(|m| m.as_str()).collect::<Vec<_>>()
    );

    let mut result = String::new();
    let mut last_end = 0;
    let mut i = 0;

    while i < token_matches.len() {
        // Attempt to merge several consecutive tokens if possible
        if let Some((merged_word, merged_count)) =
            try_merge_tokens(text, &token_matches, i, max_merge)
        {
            // If merge succeeds
            let token_start = token_matches[i].start();
            result.push_str(&text[last_end..token_start]);
            result.push_str(&merged_word);

            last_end = token_matches[i + merged_count - 1].end();
            i += merged_count;
        } else {
            // Otherwise, just copy the token
            let token_start = token_matches[i].start();
            let token_end = token_matches[i].end();

            result.push_str(&text[last_end..token_start]);
            result.push_str(&text[token_start..token_end]);

            last_end = token_end;
            i += 1;
        }
    }

    // Append the remaining text after the last token
    result.push_str(&text[last_end..]);
    result
}

/// Extracts tokens from the text (letters + apostrophes)
fn get_token_matches(text: &str) -> Vec<regex::Match<'_>> {
    // For demo purposes, we compile the regex each time
    let re = Regex::new(r"\p{L}+(?:[’']\p{L}+)*").unwrap();
    re.find_iter(text).collect()
}

/// Attempts to merge multiple consecutive tokens, starting from max_merge down to 2.
/// Returns Some((merged_word, number_of_tokens_merged)) if successful, None otherwise.
fn try_merge_tokens(
    text: &str,
    token_matches: &[regex::Match<'_>],
    start_index: usize,
    max_merge: usize,
) -> Option<(String, usize)> {
    for merge_len in (2..=max_merge).rev() {
        if start_index + merge_len <= token_matches.len() {
            // Verify that tokens are adjacent and only separated by whitespace
            if !are_tokens_adjacent(text, token_matches, start_index, merge_len) {
                continue;
            }

            // Build both the merged and spaced versions
            let (candidate, candidate_with_space, candidate_lower, candidate_with_space_lower) =
                build_candidates(text, token_matches, start_index, merge_len);

            log::info!(
                "Checking candidate: '{}' (from '{}')",
                candidate_lower,
                candidate_with_space_lower
            );

            if !is_reasonable_word(&candidate_lower) {
                log::debug!("Skipping candidate '{}': not reasonable", candidate_lower);
                continue;
            }

            // Check if the candidate exists in any DAWG
            let (in_dawg, spaced_in_dawg) =
                check_in_dawg(&candidate_lower, &candidate_with_space_lower);

            if in_dawg {
                let merge_score = compute_merge_score(&candidate_lower, merge_len);
                log::info!("Merge score for '{}': {:.2}", candidate_lower, merge_score);

                // Decide whether to merge or not
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
                log::debug!("Not merging '{}': word not found in dictionary", candidate_lower);
            }
        }
    }
    None
}

/// Verifies that tokens are consecutive and separated only by whitespace.
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

/// Builds variants (merged, spaced, lowercased, etc.)
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

    let candidate = tokens_to_merge.join("");
    let candidate_with_space = tokens_to_merge.join(" ");

    // Normalize + lowercase
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

/// Checks whether the merged word (and its spaced variant) is present in any DAWG.
fn check_in_dawg(candidate_lower: &str, candidate_with_space_lower: &str) -> (bool, bool) {
    let mut in_dawg = false;
    let mut spaced_in_dawg = false;

    for (lang, dawg) in DAWGS.0.iter() {
        if dawg_loader::contains_exact(dawg, candidate_lower) {
            log::debug!("Found '{}' in {} DAWG", candidate_lower, lang);
            in_dawg = true;
        }
        if dawg_loader::contains_exact(dawg, candidate_with_space_lower) {
            log::debug!(
                "Found spaced version '{}' in {} DAWG",
                candidate_with_space_lower,
                lang
            );
            spaced_in_dawg = true;
        }

        if let Some(word_list) = DAWGS.1.get(lang) {
            if dawg_loader::is_most_similar(word_list, candidate_lower, 1) {
                log::debug!("Found '{}' as similar in {} DAWG", candidate_lower, lang);
                in_dawg = true;
            }
        }
    }

    (in_dawg, spaced_in_dawg)
}

/// Decides whether to merge tokens, based on various conditions (score, spaced variant existence, etc.).
fn handle_merge_decision(
    candidate: &str,
    candidate_lower: &str,
    candidate_with_space: &str,
    spaced_in_dawg: bool,
    merge_len: usize,
    merge_score: f32,
) -> Option<(String, usize)> {
    // Special case: merging 2 short words (< 10 letters)
    let short_common_word = (merge_len == 2) && (candidate_lower.len() < 10);

    if short_common_word {
        let bert_score = check_word_with_bert(candidate_lower).unwrap_or(0.0);
        if spaced_in_dawg && bert_score < 0.1 {
            log::info!(
                "Not merging common short expression: '{}' (keeping '{}') [BERT score: {:.2}]",
                candidate,
                candidate_with_space,
                bert_score
            );
            None
        } else {
            log::info!(
                "Merging short word: '{}' [score: {:.2}]",
                candidate,
                merge_score
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
            log::info!(
                "Merging: '{}' [score: {:.2} >= {:.2}]",
                candidate,
                merge_score,
                threshold
            );
            Some((candidate.to_string(), merge_len))
        } else {
            log::info!(
                "Not merging: spaced version exists and score {:.2} < {:.2}",
                merge_score,
                threshold
            );
            None
        }
    }
}

/// Computes a merge score in [0..1].
fn compute_merge_score(word: &str, merge_len: usize) -> f32 {
    let len = word.len();

    // If the length is outside [3..20], return 0
    if !(3..=20).contains(&len) {
        return 0.0;
    }

    // Base score by how many tokens we're merging
    let base_score = match merge_len {
        2 => 0.50,
        3 => 0.55,
        _ => 0.60,
    };

    // Slight penalty for very short words
    let length_penalty = if len < 5 { -0.05 } else { 0.0 };

    // BERT score (approx. [0..1])
    let bert_score = match check_word_with_bert(word) {
        Ok(s) => s * 0.10, // Weighted to avoid an all-or-nothing effect
        Err(_) => {
            log::warn!("BERT check failed for '{}'", word);
            0.0
        }
    };

    let total = base_score + length_penalty + bert_score;
    total.clamp(0.0, 1.0)
}

/// Checks the plausibility of a word via BERT by computing embedding norms in multiple contexts.
/// Returns a [0..1] score.
fn check_word_with_bert(word: &str) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
    const REFERENCE_WORD: &str = "bonjour";

    // Different contexts to evaluate the word more robustly
    let contexts = [
        format!("People often use the word {}.", "{}"),
        format!("The {} is a common term in French.", "{}"),
        format!("I really like this {}.", "{}"),
        format!("He talks about {} with enthusiasm.", "{}"),
    ];

    let mut total_score = 0.0;

    // For each context, calculate a similarity-based score
    for context_template in &contexts {
        let reference_context = context_template.replace("{}", REFERENCE_WORD);
        let test_context = context_template.replace("{}", word);

        let reference_embedding = bert::encode_sentence(&reference_context)?;
        let test_embedding = bert::encode_sentence(&test_context)?;

        // L2 norm
        let reference_norm = reference_embedding
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();

        let test_norm = test_embedding
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();

        // Cosine similarity
        let dot_product: f32 = reference_embedding
            .iter()
            .zip(test_embedding.iter())
            .map(|(&a, &b)| a * b)
            .sum();

        let cosine_similarity = dot_product / (reference_norm * test_norm).max(1e-6);

        // Norm ratio used to detect anomalies
        let norm_ratio = (test_norm / reference_norm).clamp(0.0, 2.0) / 2.0;

        // Weighted combination
        let context_score = (cosine_similarity * 0.7 + norm_ratio * 0.3).clamp(0.0, 1.0);
        total_score += context_score;

        log::debug!("  - Context '{}': norm = {:.2}", test_context, test_norm);
        log::debug!(
            "  - Reference '{}': norm = {:.2}",
            reference_context,
            reference_norm
        );
        log::debug!("  - Cosine similarity: {:.2}", cosine_similarity);
        log::debug!("  - Norm ratio: {:.2}", norm_ratio);
        log::debug!("  - Context score: {:.2}", context_score);
    }

    // Final score: average across all contexts
    let combined_score = (total_score / contexts.len() as f32).clamp(0.0, 1.0);
    log::debug!("  => Final combined BERT score = {:.2}", combined_score);

    Ok(combined_score)
}
