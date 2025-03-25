pub mod actions;
mod audio;
pub mod config;
mod whisper_integration;

fn main() {
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

    // Initialisation du modèle Whisper
    let whisper_model = whisper_integration::init_model(model_path).expect("msg");

    // Chargement de la configuration utilisateur
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
    let config = config::load_config(&config_path).expect("Erreur lors du chargement de la config");

    let device = audio::get_device().expect("m");

    // Boucle principale de dictée
    loop {
        let audio_data = match audio::capture_audio(&device) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Erreur lors de la capture audio: {}", e);
                return;
            }
        };
        let transcription =
            whisper_integration::transcribe(&whisper_model, &audio_data, "fr").expect("msg");

        println!("---------------------------------------------------");
        println!("{}", &transcription);
        println!("---------------------------------------------------");

        //Analyse et mapping de la commande via le JSON
        config::execute_command(&config, transcription);
    }
}
