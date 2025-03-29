// use std::error::Error;
// use whisper_rs::WhisperState;
// use strsim::levenshtein;

// /// The keyword to detect to "wake up" the system
// //const WAKE_WORD: &str = "aurora";

// const WAKE_VARIANTS: &[&str; 10] = &[
//     "aurora",
//     "auroha",
//     "arora",
//     "auroura",
//     "uroha",
//     "laura",
//     "vox aurora",
//     "vox oroha",
//     "vox-oroha",
//     "vox au rohe."
// ];

// /// Checks if the transcription contains the magic word

// /// Minimum confidence threshold for a token (between 0.0 and 1.0)
// const TOKEN_PROB_THRESHOLD: f32 = 0.3;

// /// Checks if a segment contains a token matching the wake word
// pub fn is_wake_word_present(
//     state: &WhisperState,
//     segment_index: i32,
// ) -> Result<bool, Box<dyn Error>> {
//     let num_tokens = state.full_n_tokens(segment_index)?;
//     const LEVENSHTEIN_THRESHOLD: usize = 3;  // Allow up to 5 character differences

//     for token_index in 0..num_tokens {
//         let token_text = state.full_get_token_text(segment_index, token_index)?;
//         let token_prob = state.full_get_token_prob(segment_index, token_index)?;
//         let token_lowercase = token_text.to_lowercase();
//         let token_clean = token_lowercase.trim_matches(|c: char| !c.is_alphanumeric());

//         for &wake_word in WAKE_VARIANTS {
//             // Check for direct containment first (faster check)
//             if token_clean.contains(wake_word) && token_prob > TOKEN_PROB_THRESHOLD {
//                 println!(
//                     "✨ Wake word match: '{}' (p = {:.3})",
//                     token_clean, token_prob
//                 );
//                 return Ok(true);
//             }

//             // Check with Levenshtein distance for fuzzy matching
//             let distance = levenshtein(token_clean, wake_word);
//             if distance <= LEVENSHTEIN_THRESHOLD && token_prob > TOKEN_PROB_THRESHOLD {
//                 println!(
//                     "✨ Wake word fuzzy match: '{}' to '{}' (distance: {}, p = {:.3})",
//                     token_clean, wake_word, distance, token_prob
//                 );
//                 return Ok(true);
//             }
//         }
//     }

//     Ok(false)
// }

use crate::bert::encode_sentence;
use crate::whisper_integration;
use once_cell::sync::Lazy;
use std::error::Error;
use whisper_rs::WhisperState;

/// The wake words to detect
const WAKE_VARIANTS: &[&str; 12] = &[
    "aurora",
    "auroha",
    "arora",
    "auroura",
    "uroha",
    "laura",
    "vox aurora",
    "vox oroha",
    "vox-oroha",
    "vox au rohe.",
    "vox-orore",
    "vox ouroho.",
];

/// The minimum cosine similarity threshold to consider a match
const EMBEDDING_SIMILARITY_THRESHOLD: f32 = 0.7;

/// Pre-calculated embeddings for each wake word
static WAKE_VARIANTS_EMBEDDINGS: Lazy<Vec<Vec<f32>>> = Lazy::new(|| {
    WAKE_VARIANTS
        .iter()
        .map(|&word| {
            encode_sentence(word).unwrap_or_else(|_| {
                log::error!("Failed to encode wake word: {}", word);
                vec![]
            })
        })
        .collect()
});

/// Synchronous function that performs actual wake word detection.
fn is_wake_word_present_sync(
    state: &WhisperState,
    segment_index: i32,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    // Retrieve the raw text of the segment
    let raw_segment_text = state.full_get_segment_text(segment_index)?;
    // Clean the text using a shared function
    let segment_text = whisper_integration::clean_whisper_text(&raw_segment_text);

    // Generate the embedding from the cleaned text
    let segment_embedding = crate::bert::encode_sentence(&segment_text)?;

    for (i, &wake_word) in WAKE_VARIANTS.iter().enumerate() {
        let candidate_embedding = &WAKE_VARIANTS_EMBEDDINGS[i];
        if candidate_embedding.is_empty() {
            continue;
        }
        let similarity = crate::bert::cosine_similarity(&segment_embedding, candidate_embedding);

        log::info!(
            "Comparing cleaned segment '{}' with '{}': similarity = {:.3}",
            segment_text,
            wake_word,
            similarity
        );

        if similarity > EMBEDDING_SIMILARITY_THRESHOLD {
            log::info!("Wake word detected!");
            return Ok(true);
        }
    }

    Ok(false)
}

use std::sync::Arc;

/// Asynchronous wrapper that executes the blocking detection on a dedicated thread.
pub async fn is_wake_word_present(
    state: Arc<WhisperState>,
    segment_index: i32,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    // Move the Arc into the blocking task
    let result = tokio::task::spawn_blocking(move || {
        is_wake_word_present_sync(&state, segment_index)
    })
    .await??;
    Ok(result)
}

