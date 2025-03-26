use cpal::Device;
use cpal::traits::{DeviceTrait, HostTrait};
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

                        // As soon as enough samples are accumulated, send a chunk for processing
                        if buffer.len() > 4096 {
                            let chunk = buffer.clone();
                            buffer.clear();

                            // Use try_send instead of spawning a task
                            if let Err(e) = sender.try_send(chunk) {
                                // Check if the error is because the channel is full
                                match e {
                                    tokio::sync::mpsc::error::TrySendError::Full(_) => {
                                        eprintln!(
                                            "Audio processing channel is full, dropping sample"
                                        );
                                    }
                                    _ => {
                                        eprintln!("Channel closed: {}", e);
                                    }
                                }
                            }
                        }
                    }
                },
                |err| eprintln!("Stream error: {}", err),
                None,
            )?,
            _ => return Err("Unsupported sample format".into()),
        };

        Ok(())
    }

    // pub async fn stop_capture(&self) -> Result<(), Box<dyn Error>> {
    //     if let Some(tx) = self.keep_alive_tx.lock().unwrap().take() {
    //         let _ = tx.send(());
    //         println!("🛑 Stopping audio capture");
    //         Ok(())
    //     } else {
    //         Err("Audio capture is not active or already stopped".into())
    //     }
    // }

    pub async fn get_next_speech_segment(&mut self) -> Result<Vec<f32>, Box<dyn Error>> {
        // Get the number of channels from the default config
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
                    println!("🔊 Speech detected");
                }
                silence_start = Instant::now();
                speech_buffer.extend_from_slice(&chunk);
            } else if is_speech_active {
                speech_buffer.extend_from_slice(&chunk);
                if silence_start.elapsed() > SILENCE_DURATION_TO_FINALIZE {
                    println!("🔇 Speech segment complete");
                    let resampled = resample_to_16k(&speech_buffer, channels);
                    return Ok(resampled);
                }
            }

            if is_speech_active && speech_start.elapsed() > MAX_SPEECH_DURATION {
                println!("⏱️ Maximum speech duration reached");
                let resampled = resample_to_16k(&speech_buffer, channels);
                return Ok(resampled);
            }
        }

        Err("Audio stream ended unexpectedly".into())
    }
}

pub fn get_device() -> Result<Device, Box<dyn Error>> {
    let host = cpal::default_host();
    let devices = host.input_devices()?;
    println!("Available input devices:");
    for (i, device) in devices.enumerate() {
        println!("{}: {}", i, device.name()?);
    }

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
            host.default_input_device().ok_or("No input device found")?
        }
    };

    println!("Using device: {}", device.name()?);
    Ok(device)
}

fn resample_to_16k(input: &[f32], channels: usize) -> Vec<f32> {
    // Downmix to mono
    let mono_input: Vec<f32> = input
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();

    // Create a resampler for mono (1 channel) with chunk size 1323
    let mut resampler = rubato::FftFixedInOut::<f32>::new(44100, 16000, 1323, 1)
        .expect("Error creating resampler");

    let mut output = Vec::new();

    // Process each 1323-sample chunk individually
    for chunk in mono_input.chunks(1323) {
        // If the chunk is smaller than 1323, pad with zeros
        let mut frame = chunk.to_vec();
        if frame.len() < 1323 {
            frame.resize(1323, 0.0);
        }
        // Process a single channel input: the input slice length must equal 1 channel.
        let res = resampler
            .process(&[&frame[..]], None)
            .expect("Resampling failed");
        // The output is a Vec of one Vec<f32>, grab the first channel and append.
        output.extend_from_slice(&res[0]);
    }
    output
}

// fn pad_segment(mut segment: Vec<f32>) -> Vec<f32> {
//     // Whisper requires at least 1 second of audio at 16kHz = 16000 samples.
//     const MIN_SAMPLES: usize = 16000;
//     if segment.len() < MIN_SAMPLES {
//         let missing = MIN_SAMPLES - segment.len();
//         segment.extend(vec![0.0; missing]);
//     }
//     segment
// }
