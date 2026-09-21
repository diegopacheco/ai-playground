use semantic_cache_proxy::cache::{RedisStore, Store};
use semantic_cache_proxy::embedder::{Embedder, OllamaEmbedder};
use semantic_cache_proxy::llm::Llm;
use semantic_cache_proxy::redis::Redis;
use semantic_cache_proxy::service::{QaService, Settings};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const THRESHOLD: f32 = 0.9;

fn redis_addr() -> String {
    std::env::var("REDIS_ADDR").unwrap_or_else(|_| "127.0.0.1:6380".into())
}

fn embedder() -> OllamaEmbedder {
    OllamaEmbedder::new(&std::env::var("OLLAMA_ADDR").unwrap_or_else(|_| "127.0.0.1:11434".into()), "nomic-embed-text")
}

fn store(namespace: &str) -> RedisStore {
    let dimensions = embedder().embed("probe").unwrap().len();
    let store = RedisStore::open(Redis::new(&redis_addr()), dimensions, namespace).unwrap();
    store.clear().unwrap();
    store
}

struct SharedLlm(Arc<AtomicUsize>);

impl Llm for SharedLlm {
    fn answer(&self, question: &str) -> Result<String, String> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(format!("answer to {question}"))
    }
}

#[test]
fn real_embeddings_put_paraphrases_above_the_threshold_and_near_misses_below_it() {
    let store = store("it_knn");
    let embedder = embedder();
    let original = "What is the capital of France?";
    store.save(original, "Paris", &embedder.embed(original).unwrap()).unwrap();
    let paraphrase = store.nearest(&embedder.embed("what's France's capital city?").unwrap()).unwrap().unwrap();
    assert_eq!(paraphrase.answer, "Paris");
    assert!(paraphrase.similarity >= THRESHOLD, "paraphrase similarity {}", paraphrase.similarity);
    let different = store.nearest(&embedder.embed("What is the capital of Germany?").unwrap()).unwrap().unwrap();
    assert!(different.similarity < THRESHOLD, "near miss similarity {}", different.similarity);
    store.clear().unwrap();
}

#[test]
fn the_full_flow_over_redis_calls_the_llm_once_for_a_question_and_its_paraphrase() {
    let calls = Arc::new(AtomicUsize::new(0));
    let settings = Settings { threshold: THRESHOLD, claude_model: "counting".into(), embed_model: "nomic-embed-text".into() };
    let service = QaService::new(Box::new(embedder()), Box::new(store("it_flow")), Box::new(SharedLlm(calls.clone())), settings);
    let first = service.ask("How do I reverse a list in Python?").unwrap();
    let second = service.ask("python: reverse a list how?").unwrap();
    let third = service.ask("How do I sort a list in Python?").unwrap();
    assert!(!first.cached && second.cached && !third.cached);
    assert_eq!(second.answer, first.answer);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let stats = service.stats().unwrap();
    assert_eq!((stats.hits, stats.misses, stats.entries), (1, 2, 2));
    let entries = service.entries().unwrap();
    assert_eq!(entries[0].question, "How do I sort a list in Python?");
    assert_eq!(entries.iter().find(|entry| entry.question == first.question).unwrap().hits, 1);
    service.clear().unwrap();
}
