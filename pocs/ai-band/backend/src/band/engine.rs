use crate::agents::runner::AgentRunner;
use crate::band::prompts;
use serde::Serialize;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize)]
pub struct MusicianOutput {
    pub musician: String,
    pub round: usize,
    pub abc_notation: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ComposeEvent {
    #[serde(rename = "thinking")]
    Thinking { musician: String, round: usize },
    #[serde(rename = "done")]
    Done { musician: String, round: usize, abc_notation: String },
    #[serde(rename = "final")]
    Final { final_song: String },
    #[serde(rename = "error")]
    Error { message: String },
}

fn extract_abc(response: &str) -> String {
    let lines: Vec<&str> = response.lines().collect();
    let mut abc_lines = Vec::new();
    let mut in_abc = false;

    for line in &lines {
        let trimmed = line.trim();
        if trimmed.starts_with("X:") {
            in_abc = true;
        }
        if in_abc {
            if trimmed.starts_with("```") {
                continue;
            }
            abc_lines.push(trimmed.to_string());
        }
    }

    if abc_lines.is_empty() {
        response.trim().to_string()
    } else {
        abc_lines.join("\n")
    }
}

async fn run_musician(name: &str, prompt: String) -> Result<String, String> {
    let runner = AgentRunner::new(name);
    let response = runner.run(&prompt).await?;
    Ok(extract_abc(&response))
}

pub async fn compose_stream(genre: String, num_rounds: usize, tx: mpsc::Sender<ComposeEvent>) {
    let mut context = format!("Genre: {}\n", genre);
    let mut final_melody = String::new();
    let mut final_bass = String::new();
    let mut final_lyrics = String::new();

    for round in 1..=num_rounds {
        let _ = tx.send(ComposeEvent::Thinking { musician: "drums".into(), round }).await;
        let _ = tx.send(ComposeEvent::Thinking { musician: "bass".into(), round }).await;

        let drums_prompt = prompts::drums_prompt(round, &context);
        let bass_prompt = prompts::bass_prompt(round, &context);
        let (drums_result, bass_result) = tokio::join!(
            run_musician("drums", drums_prompt),
            run_musician("bass", bass_prompt)
        );

        let drums_abc = match drums_result {
            Ok(abc) => abc,
            Err(e) => { let _ = tx.send(ComposeEvent::Error { message: e }).await; return; }
        };
        context.push_str(&format!("\n[DRUMS Round {}]:\n{}\n", round, drums_abc));
        let _ = tx.send(ComposeEvent::Done { musician: "drums".into(), round, abc_notation: drums_abc }).await;

        let bass_abc = match bass_result {
            Ok(abc) => abc,
            Err(e) => { let _ = tx.send(ComposeEvent::Error { message: e }).await; return; }
        };
        context.push_str(&format!("\n[BASS Round {}]:\n{}\n", round, bass_abc));
        final_bass = bass_abc.clone();
        let _ = tx.send(ComposeEvent::Done { musician: "bass".into(), round, abc_notation: bass_abc }).await;

        let _ = tx.send(ComposeEvent::Thinking { musician: "melody".into(), round }).await;
        let melody_prompt = prompts::melody_prompt(round, &context);
        let melody_abc = match run_musician("melody", melody_prompt).await {
            Ok(abc) => abc,
            Err(e) => { let _ = tx.send(ComposeEvent::Error { message: e }).await; return; }
        };
        context.push_str(&format!("\n[MELODY Round {}]:\n{}\n", round, melody_abc));
        final_melody = melody_abc.clone();
        let _ = tx.send(ComposeEvent::Done { musician: "melody".into(), round, abc_notation: melody_abc }).await;

        let _ = tx.send(ComposeEvent::Thinking { musician: "lyrics".into(), round }).await;
        let lyrics_prompt = prompts::lyrics_prompt(round, &context);
        let lyrics_abc = match run_musician("lyrics", lyrics_prompt).await {
            Ok(abc) => abc,
            Err(e) => { let _ = tx.send(ComposeEvent::Error { message: e }).await; return; }
        };
        context.push_str(&format!("\n[LYRICS Round {}]:\n{}\n", round, lyrics_abc));
        final_lyrics = lyrics_abc.clone();
        let _ = tx.send(ComposeEvent::Done { musician: "lyrics".into(), round, abc_notation: lyrics_abc }).await;
    }

    let final_song = format!("{}\n{}\n{}", final_melody, final_bass, final_lyrics);
    let _ = tx.send(ComposeEvent::Final { final_song }).await;
}
