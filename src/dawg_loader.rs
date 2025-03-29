use daachorse::DoubleArrayAhoCorasick;
use std::collections::HashMap;
use std::fs;
use unicode_normalization::UnicodeNormalization;
use strsim::levenshtein;

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

// Modify your load_dawgs function to store the word lists
pub fn load_dawgs() -> (HashMap<&'static str, DoubleArrayAhoCorasick<u32>>, HashMap<&'static str, Vec<String>>) {
    // Create the target directory if it doesn't exist
    fs::create_dir_all("./dics").expect("Échec de création du répertoire ./dics");

    let mut dawgs = HashMap::new();
    let mut word_lists = HashMap::new();

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

        let dawg = DoubleArrayAhoCorasick::new(&words).expect("Échec de création du DAWG");
        dawgs.insert(*lang_code, dawg);
        word_lists.insert(*lang_code, words);
    }

    println!("🌟 Tous les DAWGs ont été construits avec succès !");
    (dawgs, word_lists)
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
        let (word, _) = line.split_once('/').unwrap_or((line, ""));
        // Normalize and lowercase the word for consistency with the candidate.
        let word = word.trim().to_lowercase().nfkc().collect::<String>();
        if !word.is_empty() && seen.insert(word.clone()) {
            words.push(word);
        }
    }

    words
}

/// Checks if a word exists exactly in the automaton.
///
/// This function verifies that `word` matches completely in the DAWG,
/// not just as a substring of longer words.
pub fn contains_exact(dawg: &DoubleArrayAhoCorasick<u32>, word: &str) -> bool {
    dawg.find_iter(word)
        .any(|m| m.start() == 0 && m.end() == word.len())
}

pub fn is_most_similar(
    word_list: &[String],
    query: &str,
    max_distance: usize
) -> bool {
    let normalized_query = query.to_lowercase().nfkc().collect::<String>();

    if let Some(min_distance) = word_list.iter()
        .map(|word| levenshtein(&normalized_query, word))
        .min() {
        println!("Distance de Levenshtein pour {}: {}", normalized_query, min_distance);
        min_distance <= max_distance
    } else {
        false
    }
}
