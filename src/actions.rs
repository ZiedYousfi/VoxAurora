use enigo::{Enigo, KeyboardControllable};
use std::error::Error;
use std::process::Command;

pub fn execute_actions(input: &str) -> Result<(), Box<dyn Error>> {
    let action = input.to_string();

    if action.starts_with("cmd:") {
        let tmp = action.strip_prefix("cmd:");
        match execute_shell_command(tmp) {
            Ok(_) => Ok(()),
            Err(e) => return Err(format!("{}", e).into()),
        }
    } else {
        let mut enigo = Enigo::new();
        match enigo.key_sequence(action) {
          Ok(_) => Ok(()),
          Err(e) => Err(format!("Failed to execute key sequence: {}", e).into()),
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
