use semantic_cache_proxy::cache::RedisStore;
use semantic_cache_proxy::config::Config;
use semantic_cache_proxy::embedder::{Embedder, OllamaEmbedder};
use semantic_cache_proxy::http::{Response, read_request, write_response};
use semantic_cache_proxy::json_out::{object, quote};
use semantic_cache_proxy::llm::ClaudeLlm;
use semantic_cache_proxy::redis::Redis;
use semantic_cache_proxy::routes::handle;
use semantic_cache_proxy::service::{QaService, Settings};
use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

fn main() {
    let config = Config::from_env();
    let embedder = OllamaEmbedder::new(&config.ollama, &config.embed_model);
    let dimensions = embedder.embed("dimension probe").unwrap_or_else(|error| fail(&error)).len();
    let store = RedisStore::open(Redis::new(&config.redis), dimensions, "qa").unwrap_or_else(|error| fail(&error));
    let settings = Settings { threshold: config.threshold, claude_model: config.claude_model.clone(), embed_model: config.embed_model.clone() };
    let service = Arc::new(QaService::new(Box::new(embedder), Box::new(store), Box::new(ClaudeLlm::new(&config.claude_model)), settings));
    let listener = TcpListener::bind(("127.0.0.1", config.port)).unwrap_or_else(|error| fail(&error.to_string()));
    println!("semantic cache proxy on http://127.0.0.1:{} using {} with {} dimensions", config.port, config.claude_model, dimensions);
    for stream in listener.incoming().flatten() {
        let service = Arc::clone(&service);
        thread::spawn(move || serve(stream, &service));
    }
}

fn serve(stream: TcpStream, service: &QaService) {
    let response = match read_request(&mut BufReader::new(&stream)) {
        Ok(request) => handle(service, &request),
        Err(error) => Response::json(400, object(&[("error", quote(&error.to_string()))])),
    };
    if let Err(error) = write_response(&mut &stream, &response) {
        eprintln!("unable to write response: {error}");
    }
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}
