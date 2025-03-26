use daachorse::DoubleArrayAhoCorasick;
use std::{collections::HashMap};

const DICTIONARIES: &[(&str, &str)] = &[
    ("fr", "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/fr_FR/fr_FR.dic"),
    ("en", "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/en_US/en_US.dic"),
];

pub fn load_dawgs() -> HashMap<&'static str, DoubleArrayAhoCorasick<u32>> {
    let mut dawgs = HashMap::new();

    for (lang_code, url) in DICTIONARIES.iter() {
        println!("⏬ Téléchargement du dictionnaire pour {}...", lang_code);

        let content = download_dic(url).expect("Erreur de téléchargement");
        let words = parse_hunspell_dic(&content);

        println!("✅ {} mots extraits pour la langue {}", words.len(), lang_code);

        let dawg = DoubleArrayAhoCorasick::new(&words).expect("Échec de création du DAWG");
        dawgs.insert(*lang_code, dawg);
    }

    println!("🌟 Tous les DAWGs ont été construits avec succès !");
    dawgs
}

fn download_dic(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let body = ureq::get(url).call()?.into_string()?;
    Ok(body)
}

fn parse_hunspell_dic(content: &str) -> Vec<String> {
    content
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once('/').or_else(|| Some((line, ""))))
        .map(|(word, _)| word.trim().to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}
