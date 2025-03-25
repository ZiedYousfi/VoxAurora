mod audio;
pub mod actions;
pub mod config;
mod whisper_integration;

fn main() {
    // Initialisation du modèle Whisper
    let whisper_model =
        whisper_integration::init_model("./models/ggml-medium.bin".to_string()).expect("msg");

    // Chargement de la configuration utilisateur
    let config =
        config::load_config("config.json").expect("Erreur lors du chargement de la config");

    // Boucle principale de dictée
    //loop {
    let audio_data = match audio::capture_audio() {
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
    //}
}
