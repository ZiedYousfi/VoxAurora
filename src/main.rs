use tokio::task::LocalSet;

pub mod actions;
mod audio;
pub mod config;
mod wakeword;
mod whisper_integration;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a LocalSet to run local tasks
    let local = LocalSet::new();

    local.run_until(async {
        let mut awake = false;

        // Get the model path from terminal input or use default
        println!(
            "Enter the path to Whisper model (or press Enter for default './models/ggml-medium.bin'):"
        );
        let mut model_path = String::new();
        std::io::stdin()
            .read_line(&mut model_path)
            .expect("Failed to read input");
        let model_path = model_path.trim();

        // Use default if input is empty
        let model_path = if model_path.is_empty() {
            "./models/ggml-small.bin".to_string()
        } else {
            model_path.to_string()
        };

        println!("Loading Whisper model from: {}", model_path);

        // Initialize the Whisper model
        let whisper_model = match whisper_integration::init_model(model_path) {
            Ok(model) => model,
            Err(e) => {
                eprintln!("Error initializing Whisper model: {}", e);
                std::process::exit(1);
            }
        };

        // Load user configuration
        // Get the config path from terminal input or use default
        println!("Enter the path to config file (or press Enter for default 'config.json'):");
        let mut config_path: Vec<String> = Vec::new();
        let mut config_path_input = String::new();

        loop {

            std::io::stdin()
                .read_line(&mut config_path_input)
                .expect("Failed to read input");
            if config_path_input.trim() == "done" {
                break;
            }
            let config_path_input = config_path_input.trim().to_string();
            config_path.push(config_path_input);

        }

        // Use default if input is empty
        if config_path.is_empty() {
            config_path.push("config.json".to_string());
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

        // Start continuous audio capture in the background
        audio_processor.start_capture().await.expect("Failed to start capture");
        println!("Listening continuously. Speak to activate commands.");

        // Main dictation loop
        loop {
            // Get the next speech segment when detected
            let audio_data = match audio_processor.get_next_speech_segment().await {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("Error during audio capture: {}", e);
                    continue;
                }
            };

            // Only proceed with transcription if we have enough audio data
            if audio_data.len() < 1000 {
                continue;
            }



            // Prepare minimal parameters for analysis (similar to transcription)
            let mut wake_params = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::default());
            wake_params.set_print_special(false);
            wake_params.set_print_progress(false);
            wake_params.set_print_realtime(false);
            wake_params.set_language(Some("fr")); // or leave None as needed

            // Create and process the state for the current audio
            let mut wake_state = whisper_model.create_state().expect("msg");
            if let Err(e) = wake_state.full(wake_params, &audio_data) {
                eprintln!("Error processing audio data for wake word detection: {}", e);
                continue;
            }

            // Check for the wake word in the first segment
            match wakeword::is_wake_word_present(&wake_state, 0) {
                Ok(true) => {
                    awake = !awake;
                    println!("System is now {}", if awake { "awake" } else { "sleeping" });
                },
                Ok(false) => {},
                Err(e) => eprintln!("Error during wake word detection: {}", e),
            }


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


            if awake {
                // Analyze and map the command via the JSON
                match config::execute_command(&config, transcription) {
                    Ok(_) => println!("Command execution completed"),
                    Err(e) => {
                        eprintln!("Failed to execute command: {}", e);
                        continue;
                    }
                };
            }
        }
    }).await;

    Ok(())
}
