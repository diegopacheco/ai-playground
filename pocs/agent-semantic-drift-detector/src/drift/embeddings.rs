use std::collections::HashMap;

pub fn text_to_embedding(text: &str) -> Vec<f64> {
    let words = tokenize(text);
    let vocab = build_vocab(&words);
    let total = words.len() as f64;

    let mut freq_map: HashMap<String, f64> = HashMap::new();
    for w in &words {
        *freq_map.entry(w.clone()).or_insert(0.0) += 1.0;
    }

    let mut embedding = vec![0.0; vocab.len()];
    for (i, term) in vocab.iter().enumerate() {
        let tf = freq_map.get(term).unwrap_or(&0.0) / total.max(1.0);
        embedding[i] = tf;
    }
    embedding
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let max_len = a.len().max(b.len());
    let mut dot = 0.0;
    let mut mag_a = 0.0;
    let mut mag_b = 0.0;

    for i in 0..max_len {
        let va = if i < a.len() { a[i] } else { 0.0 };
        let vb = if i < b.len() { b[i] } else { 0.0 };
        dot += va * vb;
        mag_a += va * va;
        mag_b += vb * vb;
    }

    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    dot / denom
}

pub fn shared_vocab_embedding(text: &str, vocab: &[String]) -> Vec<f64> {
    let words = tokenize(text);
    let total = words.len() as f64;

    let mut freq_map: HashMap<String, f64> = HashMap::new();
    for w in &words {
        *freq_map.entry(w.clone()).or_insert(0.0) += 1.0;
    }

    vocab.iter().map(|term| {
        freq_map.get(term).unwrap_or(&0.0) / total.max(1.0)
    }).collect()
}

pub fn build_shared_vocab(texts: &[&str]) -> Vec<String> {
    let mut all_words: Vec<String> = Vec::new();
    for text in texts {
        for w in tokenize(text) {
            if !all_words.contains(&w) {
                all_words.push(w);
            }
        }
    }
    all_words.sort();
    all_words
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 2)
        .map(|s| s.to_string())
        .collect()
}

fn build_vocab(words: &[String]) -> Vec<String> {
    let mut vocab: Vec<String> = Vec::new();
    for w in words {
        if !vocab.contains(w) {
            vocab.push(w.clone());
        }
    }
    vocab.sort();
    vocab
}
