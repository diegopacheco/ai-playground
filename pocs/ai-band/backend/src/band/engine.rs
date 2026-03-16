use crate::agents::runner::AgentRunner;
use crate::band::prompts;
use serde::Serialize;
use tokio::sync::mpsc;

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

pub async fn compose_stream(genre: String, num_rounds: usize, lyrics_theme: String, tx: mpsc::Sender<ComposeEvent>) {
    let mut context = format!("Genre: {}\n", genre);
    if !lyrics_theme.is_empty() {
        context.push_str(&format!("Lyrics theme/influence: {}\n", lyrics_theme));
    }
    let mut final_drums = String::new();
    let mut final_bass = String::new();
    let mut final_guitar = String::new();
    let mut final_melody = String::new();
    let mut final_lyrics = String::new();

    for round in 1..=num_rounds {
        let _ = tx.send(ComposeEvent::Thinking { musician: "drums".into(), round }).await;
        let _ = tx.send(ComposeEvent::Thinking { musician: "bass".into(), round }).await;

        let drums_p = prompts::drums_prompt(round, &genre, &context);
        let bass_p = prompts::bass_prompt(round, &genre, &context);
        let (drums_result, bass_result) = tokio::join!(
            run_musician("drums", drums_p),
            run_musician("bass", bass_p)
        );

        let drums_abc = match drums_result {
            Ok(abc) => abc,
            Err(e) => { let _ = tx.send(ComposeEvent::Error { message: e }).await; return; }
        };
        context.push_str(&format!("\n[DRUMS Round {}]:\n{}\n", round, drums_abc));
        final_drums = drums_abc.clone();
        let _ = tx.send(ComposeEvent::Done { musician: "drums".into(), round, abc_notation: drums_abc }).await;

        let bass_abc = match bass_result {
            Ok(abc) => abc,
            Err(e) => { let _ = tx.send(ComposeEvent::Error { message: e }).await; return; }
        };
        context.push_str(&format!("\n[BASS Round {}]:\n{}\n", round, bass_abc));
        final_bass = bass_abc.clone();
        let _ = tx.send(ComposeEvent::Done { musician: "bass".into(), round, abc_notation: bass_abc }).await;

        let _ = tx.send(ComposeEvent::Thinking { musician: "guitar".into(), round }).await;
        let guitar_p = prompts::guitar_prompt(round, &genre, &context);
        let guitar_abc = match run_musician("guitar", guitar_p).await {
            Ok(abc) => abc,
            Err(e) => { let _ = tx.send(ComposeEvent::Error { message: e }).await; return; }
        };
        context.push_str(&format!("\n[GUITAR Round {}]:\n{}\n", round, guitar_abc));
        final_guitar = guitar_abc.clone();
        let _ = tx.send(ComposeEvent::Done { musician: "guitar".into(), round, abc_notation: guitar_abc }).await;

        let _ = tx.send(ComposeEvent::Thinking { musician: "melody".into(), round }).await;
        let melody_p = prompts::melody_prompt(round, &genre, &context);
        let melody_abc = match run_musician("melody", melody_p).await {
            Ok(abc) => abc,
            Err(e) => { let _ = tx.send(ComposeEvent::Error { message: e }).await; return; }
        };
        context.push_str(&format!("\n[MELODY Round {}]:\n{}\n", round, melody_abc));
        final_melody = melody_abc.clone();
        let _ = tx.send(ComposeEvent::Done { musician: "melody".into(), round, abc_notation: melody_abc }).await;

        let _ = tx.send(ComposeEvent::Thinking { musician: "lyrics".into(), round }).await;
        let lyrics_p = prompts::lyrics_prompt(round, &genre, &context, &lyrics_theme);
        let lyrics_abc = match run_musician("lyrics", lyrics_p).await {
            Ok(abc) => abc,
            Err(e) => { let _ = tx.send(ComposeEvent::Error { message: e }).await; return; }
        };
        context.push_str(&format!("\n[LYRICS Round {}]:\n{}\n", round, lyrics_abc));
        final_lyrics = lyrics_abc.clone();
        let _ = tx.send(ComposeEvent::Done { musician: "lyrics".into(), round, abc_notation: lyrics_abc }).await;
    }

    let last_round = num_rounds;
    let _ = tx.send(ComposeEvent::Thinking { musician: "singer".into(), round: last_round }).await;
    let singer_p = prompts::singer_prompt(&genre, &context);
    let singer_abc = match run_musician("singer", singer_p).await {
        Ok(abc) => abc,
        Err(e) => { let _ = tx.send(ComposeEvent::Error { message: e }).await; return; }
    };
    let _ = tx.send(ComposeEvent::Done { musician: "singer".into(), round: last_round, abc_notation: singer_abc.clone() }).await;

    let final_song = format!(
        "{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}",
        final_drums, final_bass, final_guitar, final_melody, final_lyrics, singer_abc
    );
    let _ = tx.send(ComposeEvent::Final { final_song }).await;
}
