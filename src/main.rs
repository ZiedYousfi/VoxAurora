use tokio::task::LocalSet;

pub mod actions;
mod audio;
pub mod bert;
pub mod config;
mod wakeword;
pub mod whisper_integration;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let mut _server = whisper_integration::start_languagetool_server();

    bert::get_model();

    // Build the current-thread runtime manually
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let local = LocalSet::new();

    // Run the async block until completion using block_on
    rt.block_on(local.run_until(async {
        let mut awake = false;


        println!(
            "Enter the path to Whisper model (or press Enter for default './models/ggml-medium.bin'):"
        );
        let mut model_path = String::new();
        std::io::stdin()
            .read_line(&mut model_path)
            .expect("Failed to read input");
        let model_path = model_path.trim();

        let model_path = if model_path.is_empty() {
            "./models/ggml-small.bin".to_string()
        } else {
            model_path.to_string()
        };

        println!("Loading Whisper model from: {}", model_path);

        let whisper_model = match whisper_integration::init_model(model_path) {
            Ok(model) => model,
            Err(e) => {
                eprintln!("Error initializing Whisper model: {}", e);
                std::process::exit(1);
            }
        };

        println!("Enter the path to config file (or press Enter for default 'config.json'):");
        let mut config_path: Vec<String> = Vec::new();

        loop {
            let mut config_path_input = String::new();
            std::io::stdin()
                .read_line(&mut config_path_input)
                .expect("Failed to read input");
            if config_path_input.trim() == "done" {
                break;
            }
            let config_path_input = config_path_input.trim().to_string();
            if !config_path_input.is_empty(){
                config_path.push(config_path_input);
            }
        }

        if config_path.is_empty() {
            config_path.push("./configs/base_config.json".to_string());
        }

        println!("Loading config from: {:?}", config_path);
        let config = match config::load_config(config_path) {
            Ok(config) => config,
            Err(e) => {
                eprintln!("Error loading config: {}", e);
                std::process::exit(1);
            }
        };

        let device = audio::get_device().expect("Failed to get audio device");
        let mut audio_processor = audio::AudioProcessor::new(device);

        audio_processor.start_capture().await.expect("Failed to start capture");
        println!("Listening continuously. Speak to activate commands.");

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
                },
                Ok(false) => {

                },
                Err(e) => eprintln!("Error during wake word detection: {}", e),
            }

            if !awake{continue;}

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

    // Wait for the languagetool server process to finish
    if let Ok(exit_status) = _server.wait() {
        println!("LanguageTool server exited with status: {}", exit_status);
    }

    Ok(())

}
