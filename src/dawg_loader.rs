use daachorse::DoubleArrayAhoCorasick;
use std::collections::HashMap;
use std::fs;

const DICTIONARIES: &[(&str, &str)] = &[
    (
        "fr",
        "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/fr_FR/fr.dic",
    ),
    (
        "en",
        "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/en/en_US.dic",
    ),
];

pub fn load_dawgs() -> HashMap<&'static str, DoubleArrayAhoCorasick<u32>> {
    // Create the target directory if it doesn't exist
    fs::create_dir_all("./dics").expect("Échec de création du répertoire ./dics");

    let mut dawgs = HashMap::new();

    for (lang_code, url) in DICTIONARIES.iter() {
        let file_path = format!("./dics/{}.dic", lang_code);
        let content = if fs::metadata(&file_path).is_ok() {
            println!("📂 Utilisation du fichier cached pour {}...", lang_code);
            fs::read_to_string(&file_path).expect("Erreur lors de la lecture du fichier")
        } else {
            println!("⏬ Téléchargement du dictionnaire pour {}...", lang_code);
            let content = download_dic(url).expect("Erreur de téléchargement");
            fs::write(&file_path, &content).expect("Échec de l'écriture du fichier");
            content
        };

        let words = parse_hunspell_dic(&content);

        println!(
            "✅ {} mots extraits pour la langue {}",
            words.len(),
            lang_code
        );

        let dawg = DoubleArrayAhoCorasick::new(&words)
            .expect("Échec de création du DAWG");
        dawgs.insert(*lang_code, dawg);
    }

    println!("🌟 Tous les DAWGs ont été construits avec succès !");
    dawgs
}

fn download_dic(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let body = ureq::get(url)
        .call()
        .unwrap()
        .body_mut()
        .read_to_string()
        .unwrap();
    Ok(body)
}

fn parse_hunspell_dic(content: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut words = Vec::new();

    for line in content.lines().skip(1) {
        if let Some((word, _)) = line.split_once('/').or_else(|| Some((line, ""))) {
            let word = word.trim().to_lowercase();
            if !word.is_empty() && seen.insert(word.clone()) {
                words.push(word);
            }
        }
    }

    words
}
