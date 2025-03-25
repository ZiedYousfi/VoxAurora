use serde::Deserialize;
use std::fs;
use std::error::Error;

#[derive(Deserialize)]
pub struct Command {
    trigger: String,
    action: String,
}

#[derive(Deserialize)]
pub struct Config {
    commands: Vec<Command>,
}

pub fn load_config(path: &str) -> Result<Config, Box<dyn Error>> {
    let data = fs::read_to_string(path)?;
    let config: Config = serde_json::from_str(&data)?;
    Ok(config)
}
