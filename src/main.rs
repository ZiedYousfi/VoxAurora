pub mod actions;
mod audio;
pub mod config;
mod wakeword;
mod whisper_integration;

fn main() {
    let mut awake = true;
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
    let _config = match config::load_config(&config_path) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Error loading config: {}", e);
            std::process::exit(1);
        }
    };

    let device = audio::get_device().expect("m");

    // Main dictation loop
    loop {
        let audio_data = match audio::capture_audio(&device) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Error during audio capture: {}", e);
                return;
            }
        };
        let transcription = match whisper_integration::transcribe(&whisper_model, &audio_data, "fr")
        {
            Ok(text) => text,
            Err(e) => {
                eprintln!("Error during audio transcription: {}", e);
                continue;
            }
        };

        println!("---------------------------------------------------");
        println!("{}", &transcription);
        println!("---------------------------------------------------");

        if wakeword::is_wake_word_triggered(&transcription) {
            awake = !awake;
        }

        if awake {
            // Analyze and map the command via the JSON
            match config::execute_command(&_config, transcription) {
                Ok(_) => println!("Command executed successfully"),
                Err(e) => {
                    eprintln!("Failed to execute command: {}", e);
                    continue;
                }
            };
        }
    }
}
