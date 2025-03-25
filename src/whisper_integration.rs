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
    params.set_language(Some(lang));

    // Create a new state for this inference
    let mut state = model.create_state()?;

    // Process the audio data
    state.full(params, audio)?;

    // Get the number of segments
    let num_segments = state.full_n_segments()?;

    // Collect all transcribed text
    let mut result = String::new();

    // Iterate through segments and collect the text
    for i in 0..num_segments {
        if let Ok(segment) = state.full_get_segment_text(i) {
            result.push_str(&segment);
            result.push(' ');
        }
    }

    Ok(result.trim().to_string())
}
