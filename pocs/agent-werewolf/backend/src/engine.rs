use crate::agents;
use crate::db::Database;
use crate::models::*;
use crate::sse::Broadcaster;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_json::json;
use std::sync::Arc;

pub async fn run_game(
    db: Arc<Database>,
    broadcaster: Arc<Broadcaster>,
    game_id: String,
    agent_selections: Vec<AgentSelection>,
) {
    let mut rng = rand::rngs::StdRng::from_entropy();
    let werewolf_idx = rand::Rng::gen_range(&mut rng, 0..agent_selections.len());
    let werewolf_name = agent_selections[werewolf_idx].name.clone();

    let now = chrono::Utc::now().to_rfc3339();
    db.create_game(&game_id, &werewolf_name, &now);

    let mut players: Vec<(String, String, String, bool)> = vec![];
    for (i, sel) in agent_selections.iter().enumerate() {
        let role = if i == werewolf_idx { "werewolf" } else { "villager" };
        let agent_id = uuid::Uuid::new_v4().to_string();
        db.create_agent(&agent_id, &game_id, &sel.name, &sel.model, role);
        players.push((sel.name.clone(), sel.model.clone(), role.to_string(), true));
    }

    broadcaster.send(&game_id, "game_start", &json!({
        "game_id": game_id,
        "agents": agent_selections.iter().map(|a| &a.name).collect::<Vec<_>>(),
        "total_agents": agent_selections.len(),
    }));

    let mut round_number = 0;
    let mut eliminated_history: Vec<(String, String)> = vec![];

    loop {
        let alive_players: Vec<(String, String, String)> = players.iter()
            .filter(|(_, _, _, alive)| *alive)
            .map(|(n, m, r, _)| (n.clone(), m.clone(), r.clone()))
            .collect();

        let alive_villagers: Vec<&(String, String, String)> = alive_players.iter()
            .filter(|(_, _, r)| r == "villager")
            .collect();

        if alive_villagers.len() <= 1 {
            let now = chrono::Utc::now().to_rfc3339();
            db.end_game(&game_id, "werewolf", round_number, &now);
            broadcaster.send(&game_id, "game_over", &json!({
                "winner": "werewolf",
                "werewolf": werewolf_name,
                "deception_score": round_number,
                "agents": players.iter().map(|(n, _, r, a)| json!({"name": n, "role": r, "alive": a})).collect::<Vec<_>>(),
            }));
            break;
        }

        round_number += 1;

        let night_round_id = uuid::Uuid::new_v4().to_string();
        db.create_round(&night_round_id, &game_id, round_number, "night");

        broadcaster.send(&game_id, "night_phase", &json!({
            "round": round_number,
        }));

        let alive_names: Vec<String> = alive_players.iter().map(|(n, _, _)| n.clone()).collect();
        let villager_names: Vec<String> = alive_players.iter()
            .filter(|(_, _, r)| r == "villager")
            .map(|(n, _, _)| n.clone())
            .collect();

        let werewolf_model = alive_players.iter()
            .find(|(_, _, r)| r == "werewolf")
            .map(|(_, m, _)| m.clone())
            .unwrap_or_default();

        let night_prompt = format!(
            "You are playing Werewolf. You are the WEREWOLF.\n\
            Surviving players: {}\n\
            Villagers you can target: {}\n\
            {}\
            Choose one villager to eliminate tonight.\n\
            Respond with ONLY a JSON object, nothing else:\n\
            {{\"target\": \"<player_name>\", \"reasoning\": \"<why>\"}}",
            alive_names.join(", "),
            villager_names.join(", "),
            if !eliminated_history.is_empty() {
                format!("Previously eliminated: {}\n",
                    eliminated_history.iter().map(|(n, by)| format!("{} (by {})", n, by)).collect::<Vec<_>>().join(", "))
            } else { String::new() }
        );

        broadcaster.send(&game_id, "agent_thinking", &json!({
            "agent": werewolf_name,
            "phase": "night",
        }));

        let night_response = tokio::task::spawn_blocking({
            let wn = werewolf_name.clone();
            let wm = werewolf_model.clone();
            let np = night_prompt.clone();
            move || agents::run_agent(&wn, &wm, &np)
        }).await.unwrap();

        let night_action = parse_night_action(&night_response.output, &villager_names);
        let target = night_action.target.clone();

        let msg_id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        db.create_message(&msg_id, &night_round_id, &werewolf_name, "night_kill", &night_action.reasoning, Some(&target), Some(&night_response.output), Some(night_response.elapsed_ms), &now);
        db.update_round_elimination(&night_round_id, &target, "werewolf");
        db.kill_agent(&game_id, &target);

        for p in &mut players {
            if p.0 == target { p.3 = false; }
        }
        eliminated_history.push((target.clone(), "werewolf".to_string()));

        broadcaster.send(&game_id, "elimination", &json!({
            "agent": target,
            "by": "werewolf",
            "round": round_number,
        }));

        let alive_players: Vec<(String, String, String)> = players.iter()
            .filter(|(_, _, _, alive)| *alive)
            .map(|(n, m, r, _)| (n.clone(), m.clone(), r.clone()))
            .collect();

        let alive_villagers_count = alive_players.iter().filter(|(_, _, r)| r == "villager").count();
        if alive_villagers_count <= 1 {
            let now = chrono::Utc::now().to_rfc3339();
            db.end_game(&game_id, "werewolf", round_number, &now);
            broadcaster.send(&game_id, "game_over", &json!({
                "winner": "werewolf",
                "werewolf": werewolf_name,
                "deception_score": round_number,
                "agents": players.iter().map(|(n, _, r, a)| json!({"name": n, "role": r, "alive": a})).collect::<Vec<_>>(),
            }));
            break;
        }

        let day_round_id = uuid::Uuid::new_v4().to_string();
        db.create_round(&day_round_id, &game_id, round_number, "day");

        broadcaster.send(&game_id, "day_phase", &json!({
            "round": round_number,
            "eliminated_last_night": target,
        }));

        let alive_names: Vec<String> = alive_players.iter().map(|(n, _, _)| n.clone()).collect();
        let mut statements: Vec<(String, String, String)> = vec![];

        let mut shuffled_alive = alive_players.clone();
        shuffled_alive.shuffle(&mut rng);

        for (agent_name, agent_model, agent_role) in &shuffled_alive {
            let prev_statements = if statements.is_empty() {
                "No statements yet.".to_string()
            } else {
                statements.iter().map(|(n, s, _)| format!("{}: {}", n, s)).collect::<Vec<_>>().join("\n")
            };

            let discussion_prompt = if agent_role == "werewolf" {
                format!(
                    "You are playing Werewolf. You are the WEREWOLF. You must LIE and blend in.\n\
                    Your name: {}\n\
                    Surviving players: {}\n\
                    You eliminated {} last night. Do NOT reveal this.\n\
                    Previous statements this round:\n{}\n\
                    {}\
                    Make a statement that deflects suspicion away from you. Accuse someone else.\n\
                    Respond with ONLY a JSON object, nothing else:\n\
                    {{\"statement\": \"<your statement>\", \"suspect\": \"<who to frame>\"}}",
                    agent_name,
                    alive_names.join(", "),
                    target,
                    prev_statements,
                    if !eliminated_history.is_empty() {
                        format!("Elimination history: {}\n",
                            eliminated_history.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "))
                    } else { String::new() }
                )
            } else {
                format!(
                    "You are playing Werewolf. You are a VILLAGER.\n\
                    Your name: {}\n\
                    Surviving players: {}\n\
                    {} was eliminated last night by the werewolf.\n\
                    Previous statements this round:\n{}\n\
                    {}\
                    Make a statement - accuse someone, defend yourself, or share observations.\n\
                    Respond with ONLY a JSON object, nothing else:\n\
                    {{\"statement\": \"<your statement>\", \"suspect\": \"<who you suspect>\"}}",
                    agent_name,
                    alive_names.join(", "),
                    target,
                    prev_statements,
                    if !eliminated_history.is_empty() {
                        format!("Elimination history: {}\n",
                            eliminated_history.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "))
                    } else { String::new() }
                )
            };

            broadcaster.send(&game_id, "agent_thinking", &json!({
                "agent": agent_name,
                "phase": "discussion",
            }));

            let response = tokio::task::spawn_blocking({
                let an = agent_name.clone();
                let am = agent_model.clone();
                let dp = discussion_prompt.clone();
                move || agents::run_agent(&an, &am, &dp)
            }).await.unwrap();

            let action = parse_discussion_action(&response.output, agent_name);

            let msg_id = uuid::Uuid::new_v4().to_string();
            let now = chrono::Utc::now().to_rfc3339();
            db.create_message(&msg_id, &day_round_id, agent_name, "discussion", &action.statement, Some(&action.suspect), Some(&response.output), Some(response.elapsed_ms), &now);

            statements.push((agent_name.clone(), action.statement.clone(), action.suspect.clone()));

            broadcaster.send(&game_id, "discussion", &json!({
                "agent": agent_name,
                "statement": action.statement,
                "suspect": action.suspect,
                "response_time_ms": response.elapsed_ms,
            }));
        }

        broadcaster.send(&game_id, "voting_phase", &json!({
            "round": round_number,
        }));

        let mut votes: Vec<(String, String, String)> = vec![];

        for (agent_name, agent_model, agent_role) in &shuffled_alive {
            let all_statements = statements.iter()
                .map(|(n, s, _)| format!("{}: {}", n, s))
                .collect::<Vec<_>>().join("\n");

            let other_players: Vec<String> = alive_names.iter()
                .filter(|n| *n != agent_name)
                .cloned()
                .collect();

            let vote_prompt = format!(
                "You are playing Werewolf as a {}.\n\
                Your name: {}\n\
                Surviving players: {}\n\
                Statements from discussion:\n{}\n\
                {}\
                Vote to eliminate one player you believe is the werewolf.\n\
                You can vote for: {}\n\
                Respond with ONLY a JSON object, nothing else:\n\
                {{\"vote\": \"<player_name>\", \"reasoning\": \"<why>\"}}",
                agent_role,
                agent_name,
                alive_names.join(", "),
                all_statements,
                if !votes.is_empty() {
                    format!("Votes so far: {}\n",
                        votes.iter().map(|(n, v, _)| format!("{} voted for {}", n, v)).collect::<Vec<_>>().join(", "))
                } else { String::new() },
                other_players.join(", "),
            );

            broadcaster.send(&game_id, "agent_thinking", &json!({
                "agent": agent_name,
                "phase": "voting",
            }));

            let response = tokio::task::spawn_blocking({
                let an = agent_name.clone();
                let am = agent_model.clone();
                let vp = vote_prompt.clone();
                move || agents::run_agent(&an, &am, &vp)
            }).await.unwrap();

            let action = parse_vote_action(&response.output, &other_players);
            let is_correct = action.vote == werewolf_name;
            db.update_agent_votes(&game_id, agent_name, is_correct);

            let msg_id = uuid::Uuid::new_v4().to_string();
            let now = chrono::Utc::now().to_rfc3339();
            db.create_message(&msg_id, &day_round_id, agent_name, "vote", &action.reasoning, Some(&action.vote), Some(&response.output), Some(response.elapsed_ms), &now);

            votes.push((agent_name.clone(), action.vote.clone(), action.reasoning.clone()));

            broadcaster.send(&game_id, "vote", &json!({
                "agent": agent_name,
                "target": action.vote,
                "reasoning": action.reasoning,
                "response_time_ms": response.elapsed_ms,
            }));
        }

        let mut vote_counts: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
        for (_, target, _) in &votes {
            *vote_counts.entry(target.clone()).or_insert(0) += 1;
        }

        let max_votes = vote_counts.values().max().copied().unwrap_or(0);
        let top_voted: Vec<String> = vote_counts.iter()
            .filter(|(_, v)| **v == max_votes)
            .map(|(k, _)| k.clone())
            .collect();

        if top_voted.len() == 1 {
            let eliminated = top_voted[0].clone();
            let eliminated_role = alive_players.iter()
                .find(|(n, _, _)| *n == eliminated)
                .map(|(_, _, r)| r.clone())
                .unwrap_or_default();

            db.update_round_elimination(&day_round_id, &eliminated, "vote");
            db.kill_agent(&game_id, &eliminated);

            for p in &mut players {
                if p.0 == eliminated { p.3 = false; }
            }
            eliminated_history.push((eliminated.clone(), "vote".to_string()));

            broadcaster.send(&game_id, "vote_result", &json!({
                "eliminated": eliminated,
                "role": eliminated_role,
                "vote_counts": vote_counts,
                "round": round_number,
            }));

            if eliminated_role == "werewolf" {
                let now = chrono::Utc::now().to_rfc3339();
                db.end_game(&game_id, "villagers", round_number, &now);
                broadcaster.send(&game_id, "game_over", &json!({
                    "winner": "villagers",
                    "werewolf": werewolf_name,
                    "deception_score": round_number,
                    "agents": players.iter().map(|(n, _, r, a)| json!({"name": n, "role": r, "alive": a})).collect::<Vec<_>>(),
                }));
                break;
            }
        } else {
            broadcaster.send(&game_id, "vote_result", &json!({
                "eliminated": null,
                "tie": true,
                "vote_counts": vote_counts,
                "round": round_number,
            }));
        }

        if round_number >= 10 {
            let now = chrono::Utc::now().to_rfc3339();
            db.end_game(&game_id, "werewolf", round_number, &now);
            broadcaster.send(&game_id, "game_over", &json!({
                "winner": "werewolf",
                "werewolf": werewolf_name,
                "deception_score": round_number,
                "agents": players.iter().map(|(n, _, r, a)| json!({"name": n, "role": r, "alive": a})).collect::<Vec<_>>(),
            }));
            break;
        }
    }
}

fn parse_night_action(output: &str, valid_targets: &[String]) -> NightAction {
    if let Some(json_str) = extract_json(output) {
        if let Ok(action) = serde_json::from_str::<NightAction>(&json_str) {
            if valid_targets.contains(&action.target) {
                return action;
            }
        }
    }
    let target = valid_targets.choose(&mut rand::rngs::StdRng::from_entropy())
        .cloned()
        .unwrap_or_default();
    NightAction { target, reasoning: "Random selection (fallback)".to_string() }
}

fn parse_discussion_action(output: &str, agent_name: &str) -> DiscussionAction {
    if let Some(json_str) = extract_json(output) {
        if let Ok(action) = serde_json::from_str::<DiscussionAction>(&json_str) {
            return action;
        }
    }
    DiscussionAction {
        statement: format!("I think we need to be more careful about who we trust."),
        suspect: agent_name.to_string(),
    }
}

fn parse_vote_action(output: &str, valid_targets: &[String]) -> VoteAction {
    if let Some(json_str) = extract_json(output) {
        if let Ok(action) = serde_json::from_str::<VoteAction>(&json_str) {
            if valid_targets.contains(&action.vote) {
                return action;
            }
        }
    }
    let target = valid_targets.choose(&mut rand::rngs::StdRng::from_entropy())
        .cloned()
        .unwrap_or_default();
    VoteAction { vote: target, reasoning: "Random vote (fallback)".to_string() }
}

fn extract_json(text: &str) -> Option<String> {
    let text = text.trim();
    if let Some(start) = text.find('{') {
        let mut depth = 0;
        let chars: Vec<char> = text[start..].chars().collect();
        for (i, ch) in chars.iter().enumerate() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(text[start..start + i + 1].to_string());
                    }
                }
                _ => {}
            }
        }
    }
    None
}
