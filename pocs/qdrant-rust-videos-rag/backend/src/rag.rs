use anyhow::Result;

use crate::ollama::Ollama;
use crate::store::{Hit, Store};

const TOP_K: u64 = 5;
const SOURCE_CHARS: usize = 800;

pub fn build_prompt(question: &str, sources: &[Hit]) -> String {
    let mut prompt = String::from(
        "You answer questions about the videos of a YouTube channel.\n\
         Answer only from the videos listed below, in one short paragraph, \
         naming the titles of the videos that cover the question.\n\
         If none of the videos is related to the question, say that you do not know.\n\nVideos:\n",
    );
    for (index, source) in sources.iter().enumerate() {
        let description: String = source.description.chars().take(SOURCE_CHARS).collect();
        prompt.push_str(&format!(
            "[{}] Title: {}\nURL: {}\nDescription: {}\n\n",
            index + 1,
            source.title,
            source.url,
            description
        ));
    }
    prompt.push_str(&format!("Question: {question}\nAnswer:"));
    prompt
}

pub async fn ask(store: &Store, ollama: &Ollama, question: &str) -> Result<(String, Vec<Hit>)> {
    let vector = ollama.embed_query(question).await?;
    let sources = store.search(vector, TOP_K, None, None).await?;
    let answer = ollama.chat(&build_prompt(question, &sources)).await?;
    Ok((answer, sources))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(title: &str, description: &str, url: &str) -> Hit {
        Hit {
            id: String::new(),
            score: 0.9,
            video_id: String::new(),
            title: title.to_string(),
            description: description.to_string(),
            url: url.to_string(),
            thumbnail: String::new(),
            duration_text: String::new(),
            views: 0,
            age_text: String::new(),
        }
    }

    #[test]
    fn prompt_carries_every_source_so_the_answer_can_be_grounded() {
        let sources = [
            source("Rust ownership", "Moves and borrows", "https://www.youtube.com/watch?v=aaa"),
            source("Zig comptime", "Compile time code", "https://www.youtube.com/watch?v=bbb"),
        ];
        let prompt = build_prompt("What is ownership?", &sources);
        for hit in &sources {
            assert!(prompt.contains(&hit.title));
            assert!(prompt.contains(&hit.description));
            assert!(prompt.contains(&hit.url));
        }
        assert!(prompt.contains("[1] Title: Rust ownership"));
        assert!(prompt.contains("[2] Title: Zig comptime"));
    }

    #[test]
    fn prompt_forbids_answers_from_outside_the_sources() {
        let prompt = build_prompt("Who won the world cup?", &[]);
        assert!(prompt.contains("Answer only from the videos listed below"));
        assert!(prompt.contains("say that you do not know"));
    }

    #[test]
    fn prompt_ends_with_the_question_so_the_model_answers_it() {
        let prompt = build_prompt("What is ownership?", &[]);
        assert!(prompt.ends_with("Question: What is ownership?\nAnswer:"));
    }

    #[test]
    fn long_descriptions_are_cut_to_fit_the_model_context() {
        let long = "x".repeat(SOURCE_CHARS * 3);
        let prompt = build_prompt("q", &[source("Long", &long, "u")]);
        assert!(prompt.contains(&"x".repeat(SOURCE_CHARS)));
        assert!(!prompt.contains(&"x".repeat(SOURCE_CHARS + 1)));
    }
}
