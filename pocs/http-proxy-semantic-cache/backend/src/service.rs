use crate::cache::{Entry, Stats, Store};
use crate::embedder::Embedder;
use crate::llm::Llm;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    pub question: String,
    pub answer: String,
    pub cached: bool,
    pub similarity: Option<f32>,
    pub matched_question: Option<String>,
    pub latency_ms: u128,
}

pub struct Settings {
    pub threshold: f32,
    pub claude_model: String,
    pub embed_model: String,
}

pub struct QaService {
    embedder: Box<dyn Embedder>,
    store: Box<dyn Store>,
    llm: Box<dyn Llm>,
    pub settings: Settings,
}

impl QaService {
    pub fn new(embedder: Box<dyn Embedder>, store: Box<dyn Store>, llm: Box<dyn Llm>, settings: Settings) -> Self {
        Self { embedder, store, llm, settings }
    }

    pub fn ask(&self, question: &str) -> Result<Answer, String> {
        let started = Instant::now();
        let question = question.trim();
        if question.is_empty() {
            return Err("question is required".into());
        }
        let vector = self.embedder.embed(question)?;
        let nearest = self.store.nearest(&vector)?;
        if let Some(hit) = nearest.as_ref().filter(|hit| hit.similarity >= self.settings.threshold) {
            self.store.record_hit(&hit.key)?;
            return Ok(Answer {
                question: question.into(),
                answer: hit.answer.clone(),
                cached: true,
                similarity: Some(hit.similarity),
                matched_question: Some(hit.question.clone()),
                latency_ms: started.elapsed().as_millis(),
            });
        }
        let answer = self.llm.answer(question)?;
        self.store.save(question, &answer, &vector)?;
        self.store.record_miss()?;
        Ok(Answer {
            question: question.into(),
            answer,
            cached: false,
            similarity: nearest.as_ref().map(|hit| hit.similarity),
            matched_question: nearest.map(|hit| hit.question),
            latency_ms: started.elapsed().as_millis(),
        })
    }

    pub fn entries(&self) -> Result<Vec<Entry>, String> {
        self.store.entries()
    }

    pub fn stats(&self) -> Result<Stats, String> {
        self.store.stats()
    }

    pub fn clear(&self) -> Result<(), String> {
        self.store.clear()
    }
}

#[cfg(test)]
pub mod fakes {
    use super::*;
    use crate::cache::Hit;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct TableEmbedder(pub Vec<(&'static str, Vec<f32>)>);

    impl Embedder for TableEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
            self.0.iter().find(|(key, _)| *key == text).map(|(_, vector)| vector.clone()).ok_or_else(|| format!("no vector for {text}"))
        }
    }

    #[derive(Default)]
    pub struct MemoryStore {
        pub rows: Mutex<Vec<(String, String, Vec<f32>, i64)>>,
        pub hits: AtomicUsize,
        pub misses: AtomicUsize,
    }

    fn cosine(left: &[f32], right: &[f32]) -> f32 {
        let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
        let norm = |values: &[f32]| values.iter().map(|value| value * value).sum::<f32>().sqrt();
        dot / (norm(left) * norm(right))
    }

    impl Store for MemoryStore {
        fn nearest(&self, vector: &[f32]) -> Result<Option<Hit>, String> {
            let rows = self.rows.lock().unwrap();
            Ok(rows
                .iter()
                .map(|(question, answer, stored, _)| Hit { key: question.clone(), question: question.clone(), answer: answer.clone(), similarity: cosine(vector, stored) })
                .max_by(|a, b| a.similarity.total_cmp(&b.similarity)))
        }
        fn save(&self, question: &str, answer: &str, vector: &[f32]) -> Result<(), String> {
            self.rows.lock().unwrap().push((question.into(), answer.into(), vector.to_vec(), 0));
            Ok(())
        }
        fn record_hit(&self, key: &str) -> Result<(), String> {
            self.hits.fetch_add(1, Ordering::SeqCst);
            self.rows.lock().unwrap().iter_mut().filter(|row| row.0 == key).for_each(|row| row.3 += 1);
            Ok(())
        }
        fn record_miss(&self) -> Result<(), String> {
            self.misses.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn entries(&self) -> Result<Vec<Entry>, String> {
            Ok(self.rows.lock().unwrap().iter().map(|row| Entry { key: row.0.clone(), question: row.0.clone(), answer: row.1.clone(), hits: row.3, created_at: 0 }).collect())
        }
        fn stats(&self) -> Result<Stats, String> {
            Ok(Stats { hits: self.hits.load(Ordering::SeqCst) as i64, misses: self.misses.load(Ordering::SeqCst) as i64, entries: self.rows.lock().unwrap().len() as i64 })
        }
        fn clear(&self) -> Result<(), String> {
            self.rows.lock().unwrap().clear();
            Ok(())
        }
    }

    pub struct CountingLlm {
        pub calls: AtomicUsize,
        pub fail: bool,
    }

    impl Llm for CountingLlm {
        fn answer(&self, question: &str) -> Result<String, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err("claude is down".into());
            }
            Ok(format!("answer to {question}"))
        }
    }

    pub fn service(fail: bool) -> QaService {
        let embedder = TableEmbedder(vec![
            ("What is Rust?", vec![1.0, 0.0, 0.0]),
            ("what is rust", vec![0.99, 0.05, 0.0]),
            ("How tall is Everest?", vec![0.0, 1.0, 0.0]),
            ("Edge", vec![0.9, 0.43589, 0.0]),
        ]);
        let settings = Settings { threshold: 0.9, claude_model: "m".into(), embed_model: "e".into() };
        QaService::new(Box::new(embedder), Box::new(MemoryStore::default()), Box::new(CountingLlm { calls: AtomicUsize::new(0), fail }), settings)
    }
}

#[cfg(test)]
mod tests {
    use super::fakes::service;
    use super::*;

    #[test]
    fn a_paraphrase_is_served_from_cache_so_claude_is_paid_only_once() {
        let service = service(false);
        let first = service.ask("What is Rust?").unwrap();
        let second = service.ask("what is rust").unwrap();
        assert!(!first.cached);
        assert!(second.cached);
        assert_eq!(second.answer, first.answer);
        assert_eq!(second.matched_question.as_deref(), Some("What is Rust?"));
        assert_eq!(service.stats().unwrap(), Stats { hits: 1, misses: 1, entries: 1 });
    }

    #[test]
    fn an_unrelated_question_is_never_answered_with_someone_elses_answer() {
        let service = service(false);
        service.ask("What is Rust?").unwrap();
        let other = service.ask("How tall is Everest?").unwrap();
        assert!(!other.cached);
        assert_eq!(other.answer, "answer to How tall is Everest?");
        assert!(other.similarity.unwrap() < 0.9);
        assert_eq!(service.stats().unwrap().entries, 2);
    }

    #[test]
    fn a_similarity_exactly_at_the_threshold_counts_as_a_hit() {
        let service = service(false);
        service.ask("What is Rust?").unwrap();
        let edge = service.ask("Edge").unwrap();
        assert!((edge.similarity.unwrap() - 0.9).abs() < 1e-4);
        assert!(edge.cached);
    }

    #[test]
    fn a_failed_claude_call_is_not_cached_so_the_error_is_not_replayed_later() {
        let service = service(true);
        assert_eq!(service.ask("What is Rust?"), Err("claude is down".into()));
        assert_eq!(service.stats().unwrap(), Stats { hits: 0, misses: 0, entries: 0 });
    }

    #[test]
    fn a_blank_question_is_rejected_before_spending_an_embedding_or_a_claude_call() {
        assert_eq!(service(false).ask("   "), Err("question is required".into()));
    }

    #[test]
    fn clearing_the_cache_forces_the_next_question_back_to_claude() {
        let service = service(false);
        service.ask("What is Rust?").unwrap();
        service.clear().unwrap();
        assert!(!service.ask("what is rust").unwrap().cached);
    }
}
