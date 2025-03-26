use std::error::Error;
use whisper_rs::WhisperState;

/// The keyword to detect to "wake up" the system
//const WAKE_WORD: &str = "aurora";

const WAKE_VARIANTS: &[&str; 10] = &[
    "aurora",
    "auroha",
    "arora",
    "auroura",
    "uroha",
    "laura",
    "vox aurora",
    "vox oroha",
    "vox-oroha",
    "vox au rohe."
];

/// Checks if the transcription contains the magic word

/// Minimum confidence threshold for a token (between 0.0 and 1.0)
const TOKEN_PROB_THRESHOLD: f32 = 0.3;

/// Checks if a segment contains a token matching the wake word
pub fn is_wake_word_present(
    state: &WhisperState,
    segment_index: i32,
) -> Result<bool, Box<dyn Error>> {
    let num_tokens = state.full_n_tokens(segment_index)?;

    for token_index in 0..num_tokens {
        let token_text = state.full_get_token_text(segment_index, token_index)?;
        let token_prob = state.full_get_token_prob(segment_index, token_index)?;
        let token_lowercase = token_text.to_lowercase();
        let token_clean = token_lowercase.trim_matches(|c: char| !c.is_alphanumeric());

        for &wake_word in WAKE_VARIANTS {
            if token_clean.contains(wake_word) && token_prob > TOKEN_PROB_THRESHOLD {
                println!(
                    "✨ Wake word match: '{}' (p = {:.3})",
                    token_clean, token_prob
                );
                return Ok(true);
            }
        }
    }

    Ok(false)
}
