use std::error::Error;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

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

    // Nettoyer les espaces en trop (doubles espaces, avant les points, etc.)
    // Par exemple :
    clean = clean.replace(" .", ".");
    clean = clean.replace(" ,", ",");
    clean = clean.replace(" !", "!");
    clean = clean.replace(" ?", "?");

    // Retirer les espaces multiples
    let re_spaces = Regex::new(r"\s+").unwrap();
    clean = re_spaces.replace_all(&clean, " ").to_string();

    clean.trim().to_string()
}
