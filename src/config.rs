use serde::Deserialize;
use std::error::Error;
use std::fs;
use std::process::Command as ProcessCommand;

#[derive(Deserialize)]
pub struct Command {
    pub trigger: String,
    pub action: String,
}

#[derive(Deserialize)]
pub struct Config {
    pub commands: Vec<Command>,
}

pub fn load_config(path: &str) -> Result<Config, Box<dyn Error>> {
    let data = fs::read_to_string(path)?;
    let config: Config = serde_json::from_str(&data)?;
    Ok(config)
}

pub fn execute_command(config: &Config, transcription: String) {
    // Récupération des mots sous forme de liste
    let words: Vec<&str> = transcription.split_whitespace().collect();

    println!("Liste de mots:");
    for (i, word) in words.iter().enumerate() {
        println!("Mot {}: {}", i, word);
        for command in &config.commands {
            if command.trigger.to_lowercase() == word.to_lowercase() {
                println!("Command trigger detected: {}", command.trigger);
                println!("Action: {}", command.action);

                // Exécution de la commande
                match actions::execute_action(&command.action) {
                    Ok(_) => println!("Command executed successfully"),
                    Err(e) => eprintln!("Failed to execute command: {}", e),
                }
            }
        }
    }
}
