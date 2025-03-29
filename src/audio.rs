use cpal::traits::{DeviceTrait, HostTrait};
use cpal::Device;
use rubato::Resampler;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

const SILENCE_THRESHOLD: f32 = 0.01;
const MAX_SPEECH_DURATION: Duration = Duration::from_secs(10);
const SILENCE_DURATION_TO_FINALIZE: Duration = Duration::from_millis(1000);

pub struct AudioProcessor {
    pub device: Device,
    sender: mpsc::Sender<Vec<f32>>,
    receiver: mpsc::Receiver<Vec<f32>>,
    // Storage for the stop signal
    keep_alive_tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
}

impl AudioProcessor {
    pub fn new(device: Device) -> Self {
        let (sender, receiver) = mpsc::channel(100);
        AudioProcessor {
            device,
            sender,
            receiver,
            keep_alive_tx: Arc::new(Mutex::new(None)),
        }
    }

    /// Starts audio capture in a non-blocking manner.
    /// Chunks of samples are gathered and sent via a channel.
    pub async fn start_capture(&self) -> Result<(), Box<dyn Error>> {
        let config = self.device.default_input_config()?;
        let sample_format = config.sample_format();
        let config = config.into();
        let sender = self.sender.clone();

        // Buffer to accumulate audio samples
        let audio_data = Arc::new(Mutex::new(Vec::new()));
        let audio_data_clone = audio_data.clone();

        // Channel to signal stream stop
        let (keep_alive_tx, _keep_alive_rx) = tokio::sync::oneshot::channel::<()>();
        {
            let mut tx_lock = self.keep_alive_tx.lock().unwrap();
            *tx_lock = Some(keep_alive_tx);
        }

        let _stream = match sample_format {
            cpal::SampleFormat::F32 => self.device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    if let Ok(mut buffer) = audio_data_clone.lock() {
                        buffer.extend_from_slice(data);

                        // Once enough samples are accumulated, send a chunk for processing
                        if buffer.len() > 4096 {
                            let chunk = buffer.clone();
                            buffer.clear();

                            // Use try_send instead of spawning a task
                            if let Err(e) = sender.try_send(chunk) {
                                match e {
                                    tokio::sync::mpsc::error::TrySendError::Full(_) => {
                                        log::warn!(
                                            "Audio processing channel is full, dropping samples"
                                        );
                                    }
                                    _ => {
                                        log::error!("Audio channel closed: {}", e);
                                    }
                                }
                            }
                        }
                    }
                },
                |err| log::error!("Stream error: {}", err),
                None,
            )?,
            _ => return Err("Unsupported sample format".into()),
        };

        Ok(())
    }

    /*
    /// Optional function to stop capturing if you ever need it.
    pub async fn stop_capture(&self) -> Result<(), Box<dyn Error>> {
        if let Some(tx) = self.keep_alive_tx.lock().unwrap().take() {
            let _ = tx.send(());
            log::info!("Stopping audio capture");
            Ok(())
        } else {
            Err("Audio capture is not active or already stopped".into())
        }
    }
    */

    /// Continuously listens for speech segments and returns them once they are complete.
    /// - If silence is detected for `SILENCE_DURATION_TO_FINALIZE`, the segment is considered done.
    /// - If the segment exceeds `MAX_SPEECH_DURATION`, it's finalized automatically.
    pub async fn get_next_speech_segment(&mut self) -> Result<Vec<f32>, Box<dyn Error>> {
        let channels = self.device.default_input_config()?.channels() as usize;
        let mut is_speech_active = false;
        let mut speech_buffer = Vec::new();
        let mut silence_start = Instant::now();
        let mut speech_start = Instant::now();

        while let Some(chunk) = self.receiver.recv().await {
            let energy = chunk.iter().map(|sample| sample.abs()).sum::<f32>() / chunk.len() as f32;

            if energy > SILENCE_THRESHOLD {
                if !is_speech_active {
                    is_speech_active = true;
                    speech_start = Instant::now();
                    log::info!("🔊 Speech detected");
                }
                silence_start = Instant::now();
                speech_buffer.extend_from_slice(&chunk);
            } else if is_speech_active {
                // We continue to accumulate samples just in case it's a brief silence
                speech_buffer.extend_from_slice(&chunk);

                if silence_start.elapsed() > SILENCE_DURATION_TO_FINALIZE {
                    log::info!("🔇 Speech segment complete");
                    let resampled = resample_to_16k(&speech_buffer, channels);
                    return Ok(resampled);
                }
            }

            if is_speech_active && speech_start.elapsed() > MAX_SPEECH_DURATION {
                log::info!("⏱️ Maximum speech duration reached");
                let resampled = resample_to_16k(&speech_buffer, channels);
                return Ok(resampled);
            }
        }

        Err("Audio stream ended unexpectedly".into())
    }
}

/// Lets the user pick a device interactively, or defaults to the system's default device.
pub fn get_device() -> Result<Device, Box<dyn Error>> {
    let host = cpal::default_host();
    let devices = host.input_devices()?;

    println!("Available input devices:");
    for (i, device) in devices.enumerate() {
        println!("{}: {}", i, device.name()?);
    }

    println!("Please enter the index of the device you want to use (or press Enter to use default):");
    let device = match std::io::stdin()
        .lines()
        .next()
        .and_then(|line| line.ok())
        .and_then(|line| line.parse::<usize>().ok())
        .and_then(|index| host.input_devices().ok()?.nth(index))
    {
        Some(device) => device,
        None => {
            println!("Invalid selection or no selection, using the default device.");
            host.default_input_device().ok_or("No input device found")?
        }
    };

    println!("Using device: {}", device.name()?);
    Ok(device)
}

/// Resamples the given audio data to 16kHz mono.
/// Uses rubato for chunked FFT-based resampling.
fn resample_to_16k(input: &[f32], channels: usize) -> Vec<f32> {
    // Downmix to mono by averaging channels
    let mono_input: Vec<f32> = input
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();

    // Create a resampler for mono (1 channel), chunk size 1323
    let mut resampler = rubato::FftFixedInOut::<f32>::new(44100, 16000, 1323, 1)
        .expect("Error creating resampler");

    let mut output = Vec::new();

    // Process each 1323-sample chunk
    for chunk in mono_input.chunks(1323) {
        let mut frame = chunk.to_vec();
        if frame.len() < 1323 {
            // Pad with zeros if not enough samples
            frame.resize(1323, 0.0);
        }
        // The resampler expects a slice of length 1323 for each channel
        let res = resampler
            .process(&[&frame[..]], None)
            .expect("Resampling failed");
        // The output is a Vec of one Vec<f32>, so we take channel 0
        output.extend_from_slice(&res[0]);
    }
    output
}
