use std::env;
use std::fs::{self, File};
use std::io::copy;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Get cargo manifest directory (root of the project) instead of OUT_DIR
  let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
  let tools_dir = Path::new(&manifest_dir).join("tools");

  // Create tools directory if it doesn't exist
  fs::create_dir_all(&tools_dir)?;

  // Download the zip file
  let url = "https://internal1.languagetool.org/snapshots/LanguageTool-latest-snapshot.zip";
  let zip_path = tools_dir.join("LanguageTool-latest-snapshot.zip");

  if !zip_path.exists() {
    println!("cargo:warning=Downloading LanguageTool...");

    // Download the file
    let mut response = reqwest::blocking::get(url)?;
    let mut dest = File::create(&zip_path)?;
    copy(&mut response, &mut dest)?;

    // Extract the zip file
    let file = File::open(&zip_path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    println!("cargo:warning=Extracting LanguageTool...");
    archive.extract(&tools_dir)?;

    println!("cargo:warning=LanguageTool downloaded and extracted successfully.");
  } else {
    println!("cargo:warning=LanguageTool archive already exists, skipping download.");
  }

  Ok(())
}
