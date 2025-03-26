use crate::actions;
use serde::Deserialize;
use std::error::Error;
use std::fs;

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

pub fn execute_command(config: &Config, transcription: String) -> Result<(), Box<dyn Error>> {
    // Get words as a list
    let words: Vec<&str> = transcription.split_whitespace().collect();

    println!("Word list:");
    let mut result = String::new();

    // Build the complete phrase
    for (i, word) in words.iter().enumerate() {
        println!("Word {}: {}", i, word);
        result.push_str(word);
        result.push(' ');
    }

    // Check if the phrase contains a trigger
    let mut command_triggered = false;
    for command in &config.commands {
        if result.to_lowercase().contains(&command.trigger.to_lowercase()) {
            command_triggered = true;
            println!("Command trigger detected: {}", command.trigger);
            println!("Action: {}", command.action);

            // Execute the command
            match actions::execute_action(&command.action) {
                Ok(_) => println!("Command executed successfully"),
                Err(e) => eprintln!("Failed to execute command: {}", e),
            }
        }
    }

    // If no trigger was detected, execute text input once
    if !command_triggered {
        if let Err(e) = actions::execute_enigo_text(result.clone()) {
            eprintln!("Failed to execute text input: {}", e);
        }
    }

    Ok(())
}
