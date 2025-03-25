/// Le mot clé à détecter pour "réveiller" le système
const WAKE_WORD: &str = "aurora";

/// Vérifie si la transcription contient le mot magique
pub fn is_wake_word_triggered(transcription: &str) -> bool {
    transcription
        .to_lowercase()
        .split_whitespace()
        .any(|word| word.contains(WAKE_WORD))
}
