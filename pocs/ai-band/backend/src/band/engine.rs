use crate::agents::runner::AgentRunner;
use crate::band::prompts;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct MusicianOutput {
    pub musician: String,
    pub round: usize,
    pub abc_notation: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompositionResult {
    pub rounds: Vec<Vec<MusicianOutput>>,
    pub final_song: String,
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

pub async fn compose(genre: &str, num_rounds: usize) -> Result<CompositionResult, String> {
    let mut all_rounds: Vec<Vec<MusicianOutput>> = Vec::new();
    let mut context = format!("Genre: {}\n", genre);

    for round in 1..=num_rounds {
        let mut round_outputs: Vec<MusicianOutput> = Vec::new();

        let drums_runner = AgentRunner::new("drums");
        let prompt = prompts::drums_prompt(round, &context);
        let drums_response = drums_runner.run(&prompt).await?;
        let drums_abc = extract_abc(&drums_response);
        context.push_str(&format!("\n[DRUMS Round {}]:\n{}\n", round, drums_abc));
        round_outputs.push(MusicianOutput {
            musician: "drums".to_string(),
            round,
            abc_notation: drums_abc,
        });

        let bass_runner = AgentRunner::new("bass");
        let prompt = prompts::bass_prompt(round, &context);
        let bass_response = bass_runner.run(&prompt).await?;
        let bass_abc = extract_abc(&bass_response);
        context.push_str(&format!("\n[BASS Round {}]:\n{}\n", round, bass_abc));
        round_outputs.push(MusicianOutput {
            musician: "bass".to_string(),
            round,
            abc_notation: bass_abc,
        });

        let melody_runner = AgentRunner::new("melody");
        let prompt = prompts::melody_prompt(round, &context);
        let melody_response = melody_runner.run(&prompt).await?;
        let melody_abc = extract_abc(&melody_response);
        context.push_str(&format!("\n[MELODY Round {}]:\n{}\n", round, melody_abc));
        round_outputs.push(MusicianOutput {
            musician: "melody".to_string(),
            round,
            abc_notation: melody_abc,
        });

        let lyrics_runner = AgentRunner::new("lyrics");
        let prompt = prompts::lyrics_prompt(round, &context);
        let lyrics_response = lyrics_runner.run(&prompt).await?;
        let lyrics_abc = extract_abc(&lyrics_response);
        context.push_str(&format!("\n[LYRICS Round {}]:\n{}\n", round, lyrics_abc));
        round_outputs.push(MusicianOutput {
            musician: "lyrics".to_string(),
            round,
            abc_notation: lyrics_abc,
        });

        all_rounds.push(round_outputs);
    }

    let last_round = all_rounds.last().unwrap();
    let mut final_song = String::new();
    for output in last_round {
        if output.musician == "melody" || output.musician == "bass" {
            final_song.push_str(&output.abc_notation);
            final_song.push('\n');
        }
        if output.musician == "lyrics" {
            final_song.push_str(&output.abc_notation);
            final_song.push('\n');
        }
    }

    Ok(CompositionResult {
        rounds: all_rounds,
        final_song,
    })
}
