use cpal::Device;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;

/// Captures an audio slice in `f32`, mono.
#[tokio::main]
pub async fn capture_audio(device: &Device) -> Result<Vec<f32>, Box<dyn Error>> {
    let config = device.default_input_config()?;

    let sample_format = config.sample_format();
    let config = config.into();

    // We'll use this to store the audio temporarily
    let audio_data: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let audio_data_clone = audio_data.clone();

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _| {
                if let Ok(mut buffer) = audio_data_clone.lock() {
                    buffer.extend_from_slice(data);
                }
            },
            err_fn,
            None,
        )?,
        _ => return Err("Unsupported sample format".into()),
    };

    stream.play()?;
    println!("🎙️ Recording 3 seconds...");

    sleep(Duration::from_secs(3)).await; // Records for 3 seconds

    // Explicitly stop the stream and ensure it's dropped
    stream.pause()?;
    drop(stream);

    // Extract the recorded data
    let result = match audio_data.lock() {
        Ok(data) => data.clone(),
        Err(e) => return Err(format!("Failed to acquire mutex lock: {}", e).into()),
    };

    Ok(result)
}

fn err_fn(err: cpal::StreamError) {
    eprintln!("Stream error: {}", err);
}

pub fn get_device() -> Result<Device, Box<dyn Error>> {
    let host = cpal::default_host();
    // Retrieve all available input devices
    let devices = host.input_devices()?;

    // Display available devices
    println!("Available input devices:");
    for (i, device) in devices.enumerate() {
        println!("{}: {}", i, device.name()?);
    }

    // Allow the user to select a device
    let device = match std::io::stdin()
        .lines()
        .next()
        .and_then(|line| line.ok())
        .and_then(|line| line.parse::<usize>().ok())
        .and_then(|index| host.input_devices().ok()?.nth(index))
    {
        Some(device) => device,
        None => {
            println!("Invalid selection, using the default device.");
            host.default_input_device()
                .ok_or("No input device found")?
        }
    };

    println!("Using device: {}", device.name()?);

    Ok(device)
}
