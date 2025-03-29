use once_cell::sync::Lazy;
use std::collections::HashMap;

pub mod actions;
mod audio;
pub mod bert;
pub mod config;
pub mod dawg_loader;
mod wakeword;
pub mod whisper_integration;

pub static DAWGS: Lazy<(
    HashMap<&'static str, daachorse::DoubleArrayAhoCorasick<u32>>,
    HashMap<&'static str, Vec<String>>,
)> = Lazy::new(|| dawg_loader::load_dawgs());

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Chargement des DAWGS... ({} entrées)", DAWGS.0.len());

    let mut _server = whisper_integration::start_languagetool_server();
    bert::get_model();

    // Build the current-thread runtime manually
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let local = tokio::task::LocalSet::new();

    // Récupérer les arguments de la ligne de commande
    let args: Vec<String> = std::env::args().collect();

    rt.block_on(local.run_until(async move {
        // Si l'utilisateur fournit le chemin du modèle dans les arguments (après le nom du programme), on l'utilise.
        let model_path_input = if args.len() > 1 {
            args[1].clone()
        } else {
            println!("Enter the path to Whisper model (or press Enter for default './models/ggml-medium.bin'):");
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

        println!("Loading Whisper model from: {}", model_path);

        let whisper_model = match whisper_integration::init_model(model_path) {
            Ok(model) => model,
            Err(e) => {
                eprintln!("Error initializing Whisper model: {}", e);
                std::process::exit(1);
            }
        };

        // Pour la config, si des arguments supplémentaires sont fournis, on les utilise.
        let config_paths: Vec<String> = if args.len() > 2 {
            args[2..].to_vec()
        } else {
            println!("Enter the path to config file (enter paths one by one and type 'done' when finished):");
            let mut paths = Vec::new();
            loop {
                let mut line = String::new();
                std::io::stdin()
                    .read_line(&mut line)
                    .expect("Failed to read input");
                let trimmed = line.trim();
                if trimmed == "done" {
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

        println!("Loading config from: {:?}", config_paths);
        let config = match config::load_config(config_paths) {
            Ok(config) => config,
            Err(e) => {
                eprintln!("Error loading config: {}", e);
                std::process::exit(1);
            }
        };

        let device = audio::get_device().expect("Failed to get audio device");
        let mut audio_processor = audio::AudioProcessor::new(device);

        audio_processor
            .start_capture()
            .await
            .expect("Failed to start capture");
        println!("Listening continuously. Speak to activate commands.");

        // Boucle principale de traitement audio
        let mut awake = false;
        loop {
            let audio_data = match audio_processor.get_next_speech_segment().await {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("Error during audio capture: {}", e);
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

            let mut wake_state = whisper_model.create_state().expect("msg");
            if let Err(e) = wake_state.full(wake_params, &audio_data) {
                eprintln!("Error processing audio data for wake word detection: {}", e);
                continue;
            }

            match wakeword::is_wake_word_present(std::sync::Arc::new(wake_state), 0).await {
                Ok(true) => {
                    awake = !awake;
                }
                Ok(false) => {}
                Err(e) => eprintln!("Error during wake word detection: {}", e),
            }

            if !awake {
                continue;
            }

            println!("System is now {}", if awake { "awake" } else { "sleeping" });

            let transcription = match whisper_integration::transcribe(&whisper_model, &audio_data, "fr").await {
                Ok(text) => text,
                Err(e) => {
                    eprintln!("Error during audio transcription: {}", e);
                    continue;
                }
            };

            if transcription.is_empty() {
                continue;
            }

            println!("---------------------------------------------------");
            println!("{}", &transcription);
            println!("---------------------------------------------------");

            match config::execute_command(&config, transcription).await {
                Ok(_) => println!("Command execution completed"),
                Err(e) => {
                    eprintln!("Failed to execute command: {}", e);
                    continue;
                }
            };
        }
    }));

    // Attendre la fin du serveur LanguageTool
    if let Ok(exit_status) = _server.wait() {
        println!("LanguageTool server exited with status: {}", exit_status);
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
        // Start the LanguageTool server (make sure it's not already running on port 8081)
        let _server = whisper_integration::start_languagetool_server();
        // Give the server a moment to really start
        thread::sleep(Duration::from_secs(1));

        let text = "bonjour, com ment ça va ?";
        let result = whisper_integration::clean_whisper_text(text);
        // Expect the cleaned text to have fixed spacing and capitalization.
        assert_eq!(result, "Bonjour, comment ça va ?");
    }

    // Test that verifies tag removal and extra space cleanup.
    #[test]
    fn test_clean_whisper_text_removes_tags_and_extra_spaces() {
        // Input text containing special tags and multiple spaces.
        let text =
            "Voici un exemple [_BEG_]avec des [_TT_99]balises   et   des espaces   inutiles.";
        // Start the LanguageTool server if needed.
        let _server = whisper_integration::start_languagetool_server();
        thread::sleep(Duration::from_secs(1));

        let cleaned = whisper_integration::clean_whisper_text(text);
        // Verify that no special tags remain.
        assert!(!cleaned.contains("[_BEG_]"));
        assert!(!cleaned.contains("[_TT_"));
        // Verify that extra spaces are reduced.
        assert!(!cleaned.contains("  "));
    }

    // Test the merging of incorrectly separated words.
    #[test]
    fn test_merge_separated_words_dawg_regex() {
        // This example expects that if DAWGs contain the entry for "aujourd'hui",
        // then the separated phrase "au jour d hui" will be merged.
        let input_text = "Il est au jour d hui un bel après midi.";
        let merged = whisper_integration::merge_separated_words_dawg_regex(input_text, 4);
        // Check that the merged text now contains "aujourd'hui".
        // Depending on DAWG configuration, the merge might not occur if the entry is missing.
        // Assert either the merge happened.
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
        // Check that multiple spaces are reduced.
        assert!(!cleaned.contains("  "));
        // Check that stray spaces before punctuation are corrected.
        assert!(!cleaned.contains(" ,"));
        // Since LanguageTool might alter punctuation, at least ensure basic cleanup.
        assert!(cleaned.contains("Bonjour"));
    }
}
