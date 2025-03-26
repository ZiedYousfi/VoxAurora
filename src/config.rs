use crate::actions;
use serde::Deserialize;
use std::error::Error;
use std::fs;

impl AsRef<str> for Command {
    fn as_ref(&self) -> &str {
        &self.trigger
    }
}

#[derive(Deserialize, Clone)]
pub struct Command {
    pub trigger: String,
    pub action: String,
}

#[derive(Deserialize, Clone)]
pub struct Config {
    pub commands: Vec<Command>,
}

pub fn load_config(paths: Vec<String>) -> Result<Config, Box<dyn Error>> {
    let mut combined_config = Config {
        commands: Vec::new(),
    };
    let mut seen_triggers = std::collections::HashSet::new();

    for path in paths {
        match fs::read_to_string(&path) {
            Ok(data) => {
                match serde_json::from_str::<Config>(&data) {
                    Ok(config) => {
                        // Vérifie les doublons de trigger
                        for command in &config.commands {
                            let trigger_lower = command.trigger.to_lowercase();
                            if !seen_triggers.insert(trigger_lower) {
                                eprintln!("Error: Duplicate trigger found: '{}'", command.trigger);
                                panic!("Duplicate triggers are not allowed in configuration");
                            }
                        }

                        // Ajoute les commandes de ce fichier de config
                        combined_config.commands.extend(config.commands);
                        println!("Loaded config from: {}", path);
                    }
                    Err(e) => {
                        eprintln!("Error parsing config file {}: {}", path, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading config file {}: {}", path, e);
            }
        }
    }

    if combined_config.commands.is_empty() {
        return Err("No valid configuration found in any of the provided paths".into());
    }

    Ok(combined_config)
}

pub async fn execute_command(
    config: &Config,
    transcription: String,
) -> Result<(), Box<dyn std::error::Error + Send>> {
    // On délègue les opérations bloquantes
    let handle = tokio::task::spawn_blocking({
        let transcription = transcription.clone();
        let config = config.clone();
        move || -> Result<(), Box<dyn std::error::Error + Send>> {
            match crate::bert::find_best_match(&transcription, &config.commands).map_err(
                |e| -> Box<dyn std::error::Error + Send> {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("{}", e),
                    ))
                },
            )? {
                Some((command, best_score)) => {
                    println!(
                        "\u{2728} Command detected: {} (score = {:.3})",
                        command.trigger, best_score
                    );
                    match actions::execute_action(&command.action) {
                        Ok(_) => println!("Command executed successfully"),
                        Err(e) => eprintln!("Failed to execute command: {}", e),
                    }
                }
                None => {
                    println!("No matching command found. Executing raw text.");
                    if let Err(e) = actions::execute_enigo_text(transcription.clone()) {
                        eprintln!("Failed to execute text input: {}", e);
                    }
                }
            }
            Ok(())
        }
    });

    // Attente de la fin de l'opération bloquante.
    handle.await.map_err(|e| {
        Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Task join error: {}", e),
        )) as Box<dyn std::error::Error + Send>
    })?
}
