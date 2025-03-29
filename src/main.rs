use VoxAurora::{
    audio,
    bert,
    //actions,
    config,
    wakeword,
    whisper_integration,
    whisper_integration::DAWGS,
};

// On importe notre logger
mod logger;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialise le logger (activé seulement si la feature "with-logs" est présente)
    logger::init_logger();

    log::info!("Loading DAWGS... ({} entries)", DAWGS.0.len());

    let mut _server = whisper_integration::start_languagetool_server();
    bert::get_model();

    // Build the current-thread runtime manually
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let local = tokio::task::LocalSet::new();

    // Retrieve command-line arguments
    let args: Vec<String> = std::env::args().collect();

    rt.block_on(local.run_until(async move {
        // If the user provided a model path as the first argument, use it.
        // Otherwise, ask interactively.
        let model_path_input = if args.len() > 1 {
            args[1].clone()
        } else {
            println!("Please enter the path to the Whisper model (or press Enter for default './models/ggml-small.bin'):");
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .expect("Failed to read input");
            input.trim().to_string()
        };

        let model_path = if model_path_input.is_empty() {
            "./models/ggml-small.bin".to_string()
        } else {
            model_path_input
        };

        log::info!("Loading Whisper model from: {}", model_path);

        let whisper_model = match whisper_integration::init_model(model_path) {
            Ok(model) => model,
            Err(e) => {
                log::error!("Error initializing Whisper model: {}", e);
                std::process::exit(1);
            }
        };

        // If additional arguments are provided after the model path, use them as config paths.
        // Otherwise, ask the user interactively.
        let config_paths: Vec<String> = if args.len() > 2 {
            args[2..].to_vec()
        } else {
            println!("Please enter the path(s) to config file(s). Type 'done' when finished:");
            let mut paths = Vec::new();
            loop {
                let mut line = String::new();
                std::io::stdin()
                    .read_line(&mut line)
                    .expect("Failed to read input");
                let trimmed = line.trim();
                if trimmed.eq_ignore_ascii_case("done") {
                    break;
                }
                if !trimmed.is_empty() {
                    paths.push(trimmed.to_string());
                }
            }
            if paths.is_empty() {
                paths.push("./configs/base_config.json".to_string());
            }
            paths
        };

        log::info!("Loading config from: {:?}", config_paths);

        let config = match config::load_config(config_paths) {
            Ok(config) => config,
            Err(e) => {
                log::error!("Error loading config: {}", e);
                std::process::exit(1);
            }
        };

        let device = audio::get_device().expect("Failed to get audio device");
        let mut audio_processor = audio::AudioProcessor::new(device);

        audio_processor
            .start_capture()
            .await
            .expect("Failed to start capture");

        log::info!("Listening continuously. Speak to activate commands.");

        // Main audio processing loop
        let mut awake = false;
        loop {
            let audio_data = match audio_processor.get_next_speech_segment().await {
                Ok(data) => data,
                Err(e) => {
                    log::error!("Error during audio capture: {}", e);
                    continue;
                }
            };

            if audio_data.len() < 1000 {
                continue;
            }

            let mut wake_params = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::default());
            wake_params.set_print_special(false);
            wake_params.set_print_progress(false);
            wake_params.set_print_realtime(false);
            wake_params.set_token_timestamps(false);
            wake_params.set_language(Some("fr"));

            let mut wake_state = whisper_model.create_state().expect("Failed to create wake_state");
            if let Err(e) = wake_state.full(wake_params, &audio_data) {
                log::error!("Error processing audio data for wake word detection: {}", e);
                continue;
            }

            match wakeword::is_wake_word_present(std::sync::Arc::new(wake_state), 0).await {
                Ok(true) => {
                    awake = !awake;
                }
                Ok(false) => {}
                Err(e) => log::error!("Error during wake word detection: {}", e),
            }

            if !awake {
                continue;
            }

            log::info!("System is now {}", if awake { "awake" } else { "sleeping" });

            let transcription = match whisper_integration::transcribe(&whisper_model, &audio_data, "fr").await {
                Ok(text) => text,
                Err(e) => {
                    log::error!("Error during audio transcription: {}", e);
                    continue;
                }
            };

            if transcription.is_empty() {
                continue;
            }

            log::info!("---------------------------------------------------");
            log::info!("{}", &transcription);
            log::info!("---------------------------------------------------");

            match config::execute_command(&config, transcription).await {
                Ok(_) => log::info!("Command execution completed"),
                Err(e) => {
                    log::error!("Failed to execute command: {}", e);
                    continue;
                }
            };
        }
    }));

    // Wait for the LanguageTool server to exit
    if let Ok(exit_status) = _server.wait() {
        log::info!("LanguageTool server exited with status: {}", exit_status);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // This test is already present but kept for reference.
    #[test]
    fn test_burt_correct_text() {
        let _server = whisper_integration::start_languagetool_server();
        thread::sleep(Duration::from_secs(1));

        let text = "bonjour, com ment ça va ?";
        let result = whisper_integration::clean_whisper_text(text);
        assert_eq!(result, "Bonjour, comment ça va ?");
    }

    // Test that verifies tag removal and extra space cleanup.
    #[test]
    fn test_clean_whisper_text_removes_tags_and_extra_spaces() {
        let text =
            "Voici un exemple [_BEG_]avec des [_TT_99]balises   et   des espaces   inutiles.";
        let _server = whisper_integration::start_languagetool_server();
        thread::sleep(Duration::from_secs(1));

        let cleaned = whisper_integration::clean_whisper_text(text);
        assert!(!cleaned.contains("[_BEG_]"));
        assert!(!cleaned.contains("[_TT_"));
        assert!(!cleaned.contains("  "));
    }

    // Test the merging of incorrectly separated words.
    #[test]
    fn test_merge_separated_words_dawg_regex() {
        let input_text = "Il est au jour d hui un bel après midi.";
        let merged = whisper_integration::merge_separated_words_dawg_regex(input_text, 4);
        println!("Merged text: '{}'", merged);
        assert!(merged.contains("aujourd'hui"), "Merged text: '{}'", merged);
    }

    // Test that checks punctuation cleanup and spacing.
    #[test]
    fn test_clean_whisper_text_with_punctuation() {
        let text = "Bonjour , , je   suis?   là...";
        let _server = whisper_integration::start_languagetool_server();
        thread::sleep(Duration::from_secs(1));

        let cleaned = whisper_integration::clean_whisper_text(text);
        assert!(!cleaned.contains("  "));
        assert!(!cleaned.contains(" ,"));
        assert!(cleaned.contains("Bonjour"));
    }
}
