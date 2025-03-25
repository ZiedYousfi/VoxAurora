# VoxAurora

A voice command recognition system built in Rust that uses OpenAI's Whisper model for speech-to-text and executes custom commands.

## Features

- Real-time audio capture from selected input devices
- Speech recognition using Whisper ML model
- Customizable voice commands via JSON configuration
- Multi-language support (currently configured for French)

## Requirements

- Rust (latest stable version)
- [whisper-rs](https://github.com/tazz4843/whisper-rs) dependencies
- Whisper model file (e.g., `ggml-medium.bin`)

## Project Structure

- `src/main.rs` - Main application entry point
- `src/audio.rs` - Audio capture functionality
- `src/config.rs` - Configuration loading and command execution
- `src/whisper_integration.rs` - Integration with Whisper speech recognition

## Setup

1. Clone the repository
2. Download a Whisper model file and place it in the `models/` directory
3. Create a `config.json` file with your custom commands

## Usage

```bash
cargo run
```

The application will:
1. Initialize the Whisper model
2. Show available audio input devices
3. Capture audio for 5 seconds
4. Transcribe the speech
5. Execute any recognized commands

## Configuration

Commands are defined in `config.json`:

```json
{
  "commands": [
    {
      "trigger": "test",
      "action": "lancer_chrome"
    },
    {
      "trigger": "lancer terminal",
      "action": "ouvrir_terminal"
    }
  ]
}
```

## License

[MIT License](LICENSE)
