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
            "./models/ggml-medium.bin".to_string()
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
        let mut config_path = String::new();
        std::io::stdin()
            .read_line(&mut config_path)
            .expect("Failed to read input");
        let config_path = config_path.trim();

        // Use default if input is empty
        let config_path = if config_path.is_empty() {
            "config.json".to_string()
        } else {
            config_path.to_string()
        };

        println!("Loading config from: {}", config_path);
        let config = match config::load_config(&config_path) {
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

            if wakeword::is_wake_word_triggered(&transcription) {
                awake = !awake;
                println!("System is now {}", if awake { "awake" } else { "sleeping" });
            }

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
