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

/// Loads DAWGs for multiple languages and returns both the DAWG automata
/// and the original word lists in separate HashMaps.
pub fn load_dawgs() -> (
    HashMap<&'static str, DoubleArrayAhoCorasick<u32>>,
    HashMap<&'static str, Vec<String>>,
) {
    // Ensure the target directory exists
    fs::create_dir_all("./dics").expect("Failed to create ./dics directory");

    let mut dawgs = HashMap::new();
    let mut word_lists = HashMap::new();

    for (lang_code, url) in DICTIONARIES.iter() {
        let file_path = format!("./dics/{}.dic", lang_code);

        // Check if we already have a cached dictionary file
        let content = if fs::metadata(&file_path).is_ok() {
            log::info!("📂 Using cached dictionary file for {}...", lang_code);
            fs::read_to_string(&file_path).expect("Error reading cached file")
        } else {
            log::info!("⏬ Downloading dictionary for {}...", lang_code);
            let content = download_dic(url).expect("Dictionary download failed");
            fs::write(&file_path, &content).expect("Failed to write dictionary file");
            content
        };

        let words = parse_hunspell_dic(&content);

        log::info!(
            "✅ Extracted {} words for language {}",
            words.len(),
            lang_code
        );

        let dawg = DoubleArrayAhoCorasick::new(&words)
            .expect("Failed to build DAWG automaton");
        dawgs.insert(*lang_code, dawg);
        word_lists.insert(*lang_code, words);
    }

    log::info!("🌟 All DAWGs have been built successfully!");
    (dawgs, word_lists)
}

/// Downloads the dictionary content from the given `url`.
fn download_dic(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let body = ureq::get(url)
        .call()
        .unwrap()
        .body_mut()
        .read_to_string()
        .unwrap();
    Ok(body)
}

/// Parses a Hunspell `.dic` file content, skipping the first line (which often contains word count).
/// Normalizes and lowercases each word, returning a `Vec<String>` of unique entries.
fn parse_hunspell_dic(content: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut words = Vec::new();

    // Skip the first line (often just a number)
    for line in content.lines().skip(1) {
        let (word, _) = line.split_once('/').unwrap_or((line, ""));
        let word = word.trim().to_lowercase().nfkc().collect::<String>();
        if !word.is_empty() && seen.insert(word.clone()) {
            words.push(word);
        }
    }

    words
}

/// Checks if `word` is an exact match in the DAWG (not just a substring).
pub fn contains_exact(dawg: &DoubleArrayAhoCorasick<u32>, word: &str) -> bool {
    dawg.find_iter(word)
        .any(|m| m.start() == 0 && m.end() == word.len())
}

/// Determines if `query` is similar to at least one word in `word_list` within `max_distance`
/// using the Levenshtein distance.
pub fn is_most_similar(
    word_list: &[String],
    query: &str,
    max_distance: usize,
) -> bool {
    let normalized_query = query.to_lowercase().nfkc().collect::<String>();

    if let Some(min_distance) = word_list
        .iter()
        .map(|word| levenshtein(&normalized_query, word))
        .min()
    {
        log::info!("Levenshtein distance for {}: {}", normalized_query, min_distance);
        min_distance <= max_distance
    } else {
        false
    }
}
