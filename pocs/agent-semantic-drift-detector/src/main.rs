mod agents;
mod persistence;
mod drift;

use persistence::db;
use persistence::models::DriftRecord;
use drift::embeddings::text_to_embedding;
use drift::detector::analyze_drift;
use drift::plotter::plot_drift;
use agents::runner::call_llm;
use chrono::Utc;
use uuid::Uuid;

const DEFAULT_PROMPT: &str = "Explain what a hash map is and when you would use one. Be concise.";

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("run");

    let pool = db::init_db().await;

    match mode {
        "run" => {
            let prompt = args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_PROMPT);
            println!("Probing LLM with prompt: \"{}\"", prompt);

            match call_llm(prompt).await {
                Ok(response) => {
                    println!("Got response ({} chars)", response.len());

                    let embedding = text_to_embedding(&response);
                    let embedding_json = serde_json::to_string(&embedding).unwrap();

                    let record = DriftRecord {
                        id: Uuid::new_v4().to_string(),
                        prompt: prompt.to_string(),
                        response: response.clone(),
                        embedding_json,
                        created_at: Utc::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                    };

                    db::save_record(&pool, &record).await;
                    println!("Saved record {}", record.id);

                    let records = db::get_records_for_prompt(&pool, prompt).await;
                    let reports = analyze_drift(&records);
                    plot_drift(&reports);
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                }
            }
        }
        "report" => {
            let prompt = args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_PROMPT);
            println!("Generating drift report for: \"{}\"", prompt);

            let records = db::get_records_for_prompt(&pool, prompt).await;
            if records.is_empty() {
                println!("No records found. Run 'cargo run -- run' first.");
                return;
            }

            println!("Found {} records", records.len());
            let reports = analyze_drift(&records);
            plot_drift(&reports);
        }
        "history" => {
            let prompt = args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_PROMPT);
            let records = db::get_records_for_prompt(&pool, prompt).await;

            println!("=== Response History ===\n");
            for (i, record) in records.iter().enumerate() {
                println!("--- Record {} [{}] ---", i + 1, record.created_at);
                let preview: String = record.response.chars().take(200).collect();
                println!("{}", preview);
                if record.response.len() > 200 {
                    println!("...(truncated)");
                }
                println!();
            }
        }
        _ => {
            println!("Semantic Drift Detector");
            println!();
            println!("Usage:");
            println!("  cargo run -- run [prompt]      Probe the LLM and record response");
            println!("  cargo run -- report [prompt]   Show drift report for a prompt");
            println!("  cargo run -- history [prompt]  Show response history");
            println!();
            println!("Default prompt: \"{}\"", DEFAULT_PROMPT);
        }
    }
}
