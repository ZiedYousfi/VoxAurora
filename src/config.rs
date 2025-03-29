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

/// Loads a combined configuration from the given file paths.
/// It checks for duplicate triggers and logs errors if any file can't be read or parsed.
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
                        // Check for duplicate triggers
                        for command in &config.commands {
                            let trigger_lower = command.trigger.to_lowercase();
                            if !seen_triggers.insert(trigger_lower) {
                                log::error!("Duplicate trigger found: '{}'", command.trigger);
                                panic!("Duplicate triggers are not allowed in configuration");
                            }
                        }

                        // Append the commands from this config file
                        combined_config.commands.extend(config.commands);
                        log::info!("Loaded config from: {}", path);
                    }
                    Err(e) => {
                        log::error!("Error parsing config file {}: {}", path, e);
                    }
                }
            }
            Err(e) => {
                log::error!("Error reading config file {}: {}", path, e);
            }
        }
    }

    if combined_config.commands.is_empty() {
        return Err("No valid configuration found in any of the provided paths".into());
    }

    Ok(combined_config)
}

/// Executes a command based on the given transcription using the config's triggers.
/// If a matching command is found (above a threshold), we execute `actions::execute_action`;
/// otherwise, we fall back to `actions::execute_enigo_text`.
pub async fn execute_command(
    config: &Config,
    transcription: String,
) -> Result<(), Box<dyn std::error::Error + Send>> {
    // Delegate blocking operations to a separate thread
    let handle = tokio::task::spawn_blocking({
        let transcription = transcription.clone();
        let config = config.clone();
        move || -> Result<(), Box<dyn std::error::Error + Send>> {
            match crate::bert::find_best_match(&transcription, &config.commands).map_err(|e| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("{}", e),
                )) as Box<dyn std::error::Error + Send>
            })? {
                Some((command, best_score)) => {
                    log::info!("✨ Command detected: {} (score = {:.3})", command.trigger, best_score);
                    match actions::execute_action(&command.action) {
                        Ok(_) => log::info!("Command executed successfully"),
                        Err(e) => log::error!("Failed to execute command: {}", e),
                    }
                }
                None => {
                    log::info!("No matching command found. Executing raw text.");
                    if let Err(e) = actions::execute_enigo_text(transcription.clone()) {
                        log::error!("Failed to execute text input: {}", e);
                    }
                }
            }
            Ok(())
        }
    });

    // Await the blocking task's completion
    handle.await.map_err(|e| {
        Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Task join error: {}", e),
        )) as Box<dyn std::error::Error + Send>
    })?
}
