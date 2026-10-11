use anyhow::Result;

use crate::config::Config;
use crate::ollama::Ollama;
use crate::store::Store;
use crate::youtube::{self, Video};

const DESCRIPTION_CHARS: usize = 2000;
const BATCH: usize = 16;

pub fn document_text(video: &Video) -> String {
    let description: String = video.description.chars().take(DESCRIPTION_CHARS).collect();
    let mut text = format!("search_document: {}", video.title);
    if !description.is_empty() {
        text.push_str(&format!("\n\n{description}"));
    }
    if !video.keywords.is_empty() {
        text.push_str(&format!("\n\n{}", video.keywords.join(", ")));
    }
    text
}

pub async fn run(config: Config) -> Result<()> {
    let store = Store::connect(&config.qdrant_grpc_url)?;
    let ollama = Ollama::new(&config);
    store.ensure_collection().await?;
    let videos = youtube::fetch_videos().await?;
    anyhow::ensure!(!videos.is_empty(), "no videos found on the channel");
    let missing = videos
        .iter()
        .filter(|video| video.description.is_empty())
        .count();
    println!("fetched {} videos, {} without description", videos.len(), missing);
    for batch in videos.chunks(BATCH) {
        let texts: Vec<String> = batch.iter().map(document_text).collect();
        let vectors = ollama.embed(&texts).await?;
        store.upsert(batch, vectors).await?;
    }
    println!(
        "indexed {} videos, collection now holds {} points",
        videos.len(),
        store.count().await?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video(description: &str, keywords: &[&str]) -> Video {
        Video {
            id: "aaaaaaaaaaa".to_string(),
            title: "Rust ownership".to_string(),
            description: description.to_string(),
            keywords: keywords.iter().map(|keyword| keyword.to_string()).collect(),
            duration_text: "7:02".to_string(),
            duration_seconds: 422,
            views: 104,
            age_text: "10mo ago".to_string(),
        }
    }

    #[test]
    fn documents_carry_the_prefix_the_embedding_model_was_trained_with() {
        let text = document_text(&video("Moves and borrows", &["rust", "memory"]));
        assert_eq!(
            text,
            "search_document: Rust ownership\n\nMoves and borrows\n\nrust, memory"
        );
    }

    #[test]
    fn a_video_without_description_is_still_indexed_by_title() {
        assert_eq!(document_text(&video("", &[])), "search_document: Rust ownership");
    }

    #[test]
    fn descriptions_are_cut_to_stay_inside_the_embedding_context() {
        let text = document_text(&video(&"é".repeat(DESCRIPTION_CHARS + 500), &[]));
        assert_eq!(text.matches('é').count(), DESCRIPTION_CHARS);
    }
}
