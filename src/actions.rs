use enigo::*;
use std::error::Error;
use std::process::Command;

pub fn execute_action(input: &str) -> Result<(), Box<dyn Error>> {
    let action = input.to_string();

    if action.starts_with("cmd:") {
        let tmp = action.strip_prefix("cmd:").unwrap_or("");
        match execute_shell_command(tmp) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("{}", e).into()),
        }
    } else {
        let enigo_result = Enigo::new(&enigo::Settings::default());
        match enigo_result {
            Ok(mut enigo) => match enigo.text(&(action.clone() + " ")) {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("Failed to execute key sequence: {}", e).into()),
            },
            Err(e) => Err(format!("Failed to create Enigo instance: {}", e).into()),
        }
    }
}

pub fn execute_shell_command(action: &str) -> Result<(), Box<dyn Error>> {
    let status = Command::new("sh").arg("-c").arg(action).status()?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("Command exited with status: {}", status).into())
    }
}
