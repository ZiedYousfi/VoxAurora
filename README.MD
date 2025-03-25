# ✨ VoxAurora 🔮

A magical voice command recognition system built in Rust that uses OpenAI's Whisper model for speech-to-text and executes custom commands. This README was created using VoxAurora itself! 🎙️💫

## ✨ Features

- 🎤 Real-time audio capture from selected input devices
- 👂 Continuous speech monitoring with automatic speech detection
- 🧠 Speech recognition using Whisper ML model
- ⚙️ Customizable voice commands via JSON configuration
- 🔮 "Wake word" functionality to toggle system activity
- 🌍 Multi-language support (currently configured for French)
- ⌨️ Text input simulation and shell command execution capabilities

## 🔧 Requirements

- 🦀 Rust (latest stable version)
- 🤫 [whisper-rs](https://github.com/tazz4843/whisper-rs) dependencies
- 📚 Whisper model file (e.g., `ggml-medium.bin`)
- 🖱️ Enigo library for text input simulation
- 🎵 Rubato library for audio resampling

## 📁 Project Structure

- `src/main.rs` - ✨ Main application entry point and event loop
- `src/audio.rs` - 🎵 Audio capture and processing functionality
- `src/config.rs` - 📝 Configuration loading and command matching
- `src/actions.rs` - 🚀 Command execution (shell and text input)
- `src/whisper_integration.rs` - 🤖 Integration with Whisper speech recognition
- `src/wakeword.rs` - 🔮 Wake word detection to toggle system activity

## 🚀 Setup

1. 📥 Clone the repository
2. 🧠 Download a Whisper model file and place it in the `models/` directory
3. ⚙️ Create a `config.json` file with your custom commands
4. 📦 Install dependencies: `cargo build`

## 🎮 Usage

```bash
cargo run
```

The application will:
1. 🔍 Prompt for Whisper model path (defaults to './models/ggml-medium.bin')
2. 📋 Prompt for configuration file path (defaults to 'config.json')
3. 🎤 Show available audio input devices and prompt for selection
4. 👂 Start listening continuously for voice commands
5. ✨ Activate when the wake word "aurora" is detected
6. 🔮 Transcribe speech and execute matched commands

## ⚙️ Configuration

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
      "action": "cmd:gnome-terminal"
    }
  ]
}
```

Action types:
- 💻 Text prefixed with `cmd:` will be executed as shell commands
- ⌨️ Other text will be simulated as keyboard input

Many more features are planned for VoxAurora. Stay tuned for updates! 🌟

## 🌟 Advanced Features

- 🎭 Automatic speech detection with configurable silence thresholds
- 🎛️ Audio downsampling from 44.1kHz to 16kHz for Whisper compatibility
- 🔮 Wake word system to toggle between active and sleeping modes
- 🧙‍♀️ Support for both text input simulation and system command execution

## 💖 The Magic of VoxAurora

VoxAurora captures your voice and transforms it into digital magic! With the charm of your voice, command your computer with grace and elegance.

*"Words have power. Let me show you."*

## 📜 License

[MIT License](LICENSE) 🔓
