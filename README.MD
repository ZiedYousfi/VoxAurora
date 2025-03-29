# ✨ VoxAurora 🔮

A powerful voice command recognition system built in Rust that leverages OpenAI's Whisper model for speech-to-text transcription and executes custom commands based on natural language input. This README was created using VoxAurora itself! 🎙️💫

## ✨ Features

- 🎤 Real-time audio capture with device selection
- 👂 Intelligent speech detection with automatic silence detection
- 🧠 High-quality speech recognition using Whisper ML models
- ⚙️ Configurable voice commands via JSON files
- 🔮 "Wake word" detection with "aurora" and variants
- 🧐 Semantic command matching using BERT embeddings
- 🌍 Multilingual support (French and English dictionaries)
- ⌨️ Text input simulation for typing by voice
- 🐚 Shell command execution capabilities
- 📝 Text correction using LanguageTool
- 📚 Word merging with dictionary verification
- 🧙‍♂️ Natural language processing for command intent

## 🧠 How It Works

1. 🎤 Captures audio continuously from selected input device
2. 👂 Detects speech by analyzing audio energy levels
3. 🔮 Listens for wake word ("aurora" or variants) to activate/deactivate
4. 🤖 When active, transcribes speech using Whisper model
5. 🧐 Applies text corrections and merges incorrectly separated words
6. 🔍 Uses BERT embeddings to find the closest matching command
7. 🚀 Executes the matched command or types the transcribed text

## 🔧 Requirements

- 🦀 Rust (latest stable version)
- 🤫 [whisper-rs](https://github.com/tazz4843/whisper-rs) dependencies
- 📚 Whisper model file (e.g., `ggml-small.bin`)
- ☕ Java runtime (for LanguageTool)
- 🖱️ Enigo library for text input simulation
- 🎵 Rubato library for audio resampling
- 🤖 rust-bert for semantic command matching

## 📁 Project Structure

- `src/main.rs` - Main application entry point and runtime setup
- `src/audio.rs` - Audio capture, processing, and speech segmentation
- `src/whisper_integration.rs` - Whisper model integration with text cleanup
- `src/config.rs` - Configuration loading and command execution
- `src/actions.rs` - Shell command and keyboard input execution
- `src/bert.rs` - BERT embeddings for semantic similarity matching
- `src/wakeword.rs` - Wake word detection to toggle system activity
- `src/dawg_loader.rs` - Dictionary loading and word verification
- `build.rs` - Build script to download LanguageTool
- `configs/` - JSON configuration files for commands

## 🚀 Setup

1. 📥 Clone the repository
2. 🧠 Download a [Whisper model](https://huggingface.co/ggerganov/whisper.cpp/tree/main) and place it in the `models/` directory
3. ⚙️ Create or modify configuration files in the `configs/` directory
4. 📦 Build the project: `cargo build`

## 🎮 Usage

```bash
cargo run [path/to/whisper/model] [path/to/config1.json] [path/to/config2.json] ...
```

If you don't provide command-line arguments, the application will:
1. 🔍 Prompt for Whisper model path (defaults to './models/ggml-small.bin')
2. 📋 Prompt for configuration file paths (multiple configs supported)
3. 🎤 List available audio input devices and prompt for selection
4. 👂 Start continuous listening for voice input

## ⚙️ Configuration

Commands are defined in JSON configuration files:

```json
{
  "commands": [
    {
      "trigger": "open browser",
      "action": "cmd:google-chrome"
    },
    {
      "trigger": "open terminal",
      "action": "cmd:gnome-terminal"
    },
    {
      "trigger": "write hello world",
      "action": "Hello, World!"
    },
    {
      "trigger": "add debug",
      "action": "console.log('Debug:', );"
    }
  ]
}
```

Action types:
- 💻 Text prefixed with `cmd:` will be executed as shell commands
- ⌨️ Other text will be simulated as keyboard input

VoxAurora comes with two example configuration files:
- `configs/base_config.json` - Basic system commands
- `configs/rust_fr.json` - Rust programming commands in French

Many more features are planned for VoxAurora. Stay tuned for updates! 🌟

## 🔍 Advanced Features

### 🔮 Wake Word Detection
Say "aurora" (or variants like "vox aurora") to toggle between active and sleep modes.

### 🧠 Semantic Command Matching
Commands are matched using BERT sentence embeddings, allowing for natural variations in how commands are spoken.

### 📝 Text Correction
Transcribed text is processed through LanguageTool for grammar and spelling correction.

### 📚 Word Merging
The system intelligently merges words that may have been incorrectly separated during transcription using dictionary verification.

### 🎭 Speech Detection
Automatically detects the start and end of speech with configurable silence thresholds.

## 🛠️ Development

- Run tests with `cargo test`
- Add new commands by creating or modifying JSON configuration files
- Customize wake words in the `wakeword.rs` file

## 💖 The Magic of VoxAurora

VoxAurora captures your voice and transforms it into digital magic! With the charm of your voice, command your computer with grace and elegance.

*"Words have power. Let me show you."*

## 📜 License

[MIT License](LICENSE) 🔓
