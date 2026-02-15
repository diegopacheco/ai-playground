use attractor::{Parser, PipelineGraph, Pipeline, Stylesheet};
use llm_client::LlmClient;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: attractor-cli <pipeline.dot> [stylesheet.json]");
        std::process::exit(1);
    }

    let dot_path = &args[1];
    let dot_content = std::fs::read_to_string(dot_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", dot_path, e);
        std::process::exit(1);
    });

    let mut parser = Parser::new(&dot_content);
    let dot_graph = parser.parse().unwrap_or_else(|e| {
        eprintln!("Failed to parse DOT: {}", e);
        std::process::exit(1);
    });

    let stylesheet = if args.len() > 2 {
        let ss_content = std::fs::read_to_string(&args[2]).unwrap_or_else(|e| {
            eprintln!("Failed to read stylesheet: {}", e);
            std::process::exit(1);
        });
        Stylesheet::from_json(&ss_content).unwrap_or_else(|e| {
            eprintln!("Failed to parse stylesheet: {}", e);
            std::process::exit(1);
        })
    } else {
        Stylesheet::new()
    };

    let pipeline_graph = PipelineGraph::from_dot(&dot_graph).unwrap_or_else(|e| {
        eprintln!("Failed to build graph: {}", e);
        std::process::exit(1);
    });

    let default_model = std::env::var("ATTRACTOR_MODEL")
        .unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());

    let provider = llm_client::catalog::provider_for_model(&default_model)
        .unwrap_or("anthropic");

    let client = LlmClient::from_env(provider).unwrap_or_else(|e| {
        eprintln!("Failed to create LLM client: {}", e);
        std::process::exit(1);
    });

    let pipeline = Pipeline::new(pipeline_graph, stylesheet, client, default_model);

    match pipeline.run().await {
        Ok(state) => {
            println!("Pipeline completed successfully.");
            println!("Final state:");
            let vars = state.all_vars();
            let mut keys: Vec<&String> = vars.keys().collect();
            keys.sort();
            for key in keys {
                println!("  {} = {}", key, vars[key]);
            }
        }
        Err(e) => {
            eprintln!("Pipeline failed: {}", e);
            std::process::exit(1);
        }
    }
}
