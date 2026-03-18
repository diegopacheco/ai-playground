use std::collections::HashMap;

pub struct CosineEval;

impl CosineEval {
    pub fn score(golden: &str, candidate: &str) -> f64 {
        let golden_vec = Self::tfidf_vector(golden);
        let candidate_vec = Self::tfidf_vector(candidate);
        Self::cosine_similarity(&golden_vec, &candidate_vec)
    }

    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .map(|s| s.to_string())
            .collect()
    }

    fn term_frequency(tokens: &[String]) -> HashMap<String, f64> {
        let mut freq: HashMap<String, f64> = HashMap::new();
        let len = tokens.len() as f64;
        for token in tokens {
            *freq.entry(token.clone()).or_insert(0.0) += 1.0;
        }
        for val in freq.values_mut() {
            *val /= len;
        }
        freq
    }

    fn tfidf_vector(text: &str) -> HashMap<String, f64> {
        let tokens = Self::tokenize(text);
        Self::term_frequency(&tokens)
    }

    fn cosine_similarity(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
        let mut dot = 0.0;
        let mut mag_a = 0.0;
        let mut mag_b = 0.0;

        for (key, val) in a {
            mag_a += val * val;
            if let Some(bval) = b.get(key) {
                dot += val * bval;
            }
        }
        for val in b.values() {
            mag_b += val * val;
        }

        let magnitude = mag_a.sqrt() * mag_b.sqrt();
        if magnitude == 0.0 {
            return 0.0;
        }
        dot / magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_texts() {
        let score = CosineEval::score("the cat sat on the mat", "the cat sat on the mat");
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_similar_texts() {
        let score = CosineEval::score(
            "the cat sat on the mat",
            "a cat was sitting on the mat"
        );
        assert!(score > 0.3);
    }

    #[test]
    fn test_different_texts() {
        let score = CosineEval::score(
            "rust programming language memory safety",
            "jupiter saturn neptune orbits galaxies"
        );
        assert!(score < 0.1);
    }
}
