use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};
use std::cell::RefCell;
use std::thread_local;

thread_local! {
    /// Thread-local storage for the sentence embeddings model.
    static SENTENCE_EMBEDDINGS_MODEL: RefCell<Option<SentenceEmbeddingsModel>> = const { RefCell::new(None) };
}

/// Retrieves or initializes the global Sentence Embeddings model.
/// Once initialized, it will stay in memory for the remainder of the program.
pub fn get_model() -> &'static SentenceEmbeddingsModel {
    SENTENCE_EMBEDDINGS_MODEL.with(|model_cell| {
        let mut model_ref = model_cell.borrow_mut();
        if model_ref.is_none() {
            *model_ref = Some(
                SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                    .create_model()
                    .expect("Failed to initialize Sentence Embeddings model"),
            );
        }
        // This is safe because the RefCell lives for the entire program.
        unsafe { &*(model_ref.as_ref().unwrap() as *const SentenceEmbeddingsModel) }
    })
}

/// Encodes a single sentence into a vector of floats.
pub fn encode_sentence(
    sentence: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    let model = get_model();
    let output = model
        .encode(&[sentence])
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
    Ok(output[0].clone())
}

/// Computes the cosine similarity between two float slices.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

/// Finds the best match in `candidates` for the given `input` string, if any.
/// Returns an `Option` containing `(best_candidate, similarity_score)`.
pub fn find_best_match<T: AsRef<str> + Clone>(
    input: &str,
    candidates: &[T],
) -> Result<Option<(T, f32)>, Box<dyn std::error::Error + Send + Sync>> {
    let input_embedding = encode_sentence(input)?;
    let threshold = 0.75;
    let mut best_score = 0.0;
    let mut best_candidate: Option<T> = None;

    for candidate in candidates {
        let candidate_str = candidate.as_ref();
        let candidate_embedding = encode_sentence(candidate_str)?;
        let similarity = cosine_similarity(&input_embedding, &candidate_embedding);

        log::info!(
            "Comparing input with candidate '{}': similarity = {:.3}",
            candidate_str,
            similarity
        );

        if similarity > threshold && similarity > best_score {
            best_score = similarity;
            best_candidate = Some(candidate.clone());
        }
    }

    Ok(best_candidate.map(|c| (c, best_score)))
}
