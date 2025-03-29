// src/logger.rs
#[cfg(feature = "with-logs")]
pub fn init_logger() {
    let _ = env_logger::try_init();
}

#[cfg(not(feature = "with-logs"))]
pub fn init_logger() {
    // Les logs sont désactivés, on fait rien
}
