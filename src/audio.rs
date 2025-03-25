use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::error::Error;
use std::time::Duration;
use tokio::time::sleep;

/// Capture une tranche d'audio en `f32`, mono.
#[tokio::main]
pub async fn capture_audio() -> Result<Vec<f32>, Box<dyn Error>> {
    let host = cpal::default_host();
    // Récupérer tous les périphériques d'entrée disponibles
    let devices = host.input_devices()?;

    // Afficher les périphériques disponibles
    println!("Périphériques d'entrée disponibles:");
    for (i, device) in devices.enumerate() {
      println!("{}: {}", i, device.name()?);
    }

    // Permettre à l'utilisateur de sélectionner un périphérique
    let device = match std::io::stdin()
      .lines()
      .next()
      .and_then(|line| line.ok())
      .and_then(|line| line.parse::<usize>().ok())
      .and_then(|index| host.input_devices().ok()?.nth(index))
    {
      Some(device) => device,
      None => {
        println!("Sélection non valide, utilisation du périphérique par défaut.");
        host.default_input_device()
          .ok_or("Aucun périphérique d'entrée trouvé")?
      }
    };

    println!("Utilisation du périphérique: {}", device.name()?);
    let config = device.default_input_config()?;

    let sample_format = config.sample_format();
    let config = config.into();

    // On utilisera ça pour stocker l'audio temporairement
    let audio_data: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let audio_data_clone = audio_data.clone();

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _| {
                let mut buffer = audio_data_clone.lock().unwrap();
                buffer.extend_from_slice(data);
            },
            err_fn,
            None,
        )?,
        _ => return Err("Unsupported sample format".into()),
    };

    stream.play()?;
    println!("🎙️ Recording 1 seconds...");

    sleep(Duration::from_secs(1)).await; // Enregistre 3 secondes
    drop(stream); // Arrête le stream

    let result = match Arc::try_unwrap(audio_data) {
      Ok(mutex) => match mutex.into_inner() {
        Ok(data) => data,
        Err(e) => return Err(format!("Failed to acquire mutex lock: {}", e).into()),
      },
      Err(_) => return Err("Failed to unwrap Arc, there are still other references".into()),
    };
    Ok(result)
}

fn err_fn(err: cpal::StreamError) {
    eprintln!("Stream error: {}", err);
}
