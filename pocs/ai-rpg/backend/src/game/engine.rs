use crate::persistence::db;
use crate::agents::claude;
use crate::sse::broadcaster::{Broadcaster, GameEvent};
use sqlx::SqlitePool;
use std::sync::Arc;

pub async fn handle_action(
    pool: &SqlitePool,
    broadcaster: &Arc<Broadcaster>,
    game_id: &str,
    player_action: &str,
) {
    db::save_message(pool, game_id, "player", player_action).await;

    broadcaster.send(game_id, GameEvent::DmThinking);

    let game = db::get_game(pool, game_id).await.unwrap();
    let character = db::get_character(pool, game_id).await.unwrap();
    let messages = db::get_messages(pool, game_id).await;

    let history = messages.iter()
        .map(|(role, content)| format!("[{}]: {}", role, content))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        r#"You are a Dungeon Master for a text-based RPG game. You must stay in character at all times.

Setting: {}
Player name: {}
Player stats: HP {}/{}, Level {}, XP {}, Gold {}
Inventory: {}
Current location: {}

Conversation history:
{}

The player just said: "{}"

Respond as the Dungeon Master. Describe what happens next in a vivid, engaging way (2-4 paragraphs).
Include any consequences of their action (combat outcomes, items found, NPCs met, etc).

IMPORTANT: At the end of your response, you MUST include a JSON block with updated character stats in this exact format:
```json
{{"hp": <number>, "xp": <number>, "gold": <number>, "level": <number>, "inventory": [<list of strings>], "location": "<current location>"}}
```
Only change stats that should change based on what happened. Keep the JSON on its own line after your narrative."#,
        game.setting,
        game.player_name,
        character.hp, character.max_hp, character.level, character.xp, character.gold,
        character.inventory,
        character.location,
        history,
        player_action
    );

    match claude::call_claude(&prompt).await {
        Ok(response) => {
            let (narrative, stats) = parse_response(&response, &character);
            db::save_message(pool, game_id, "dm", &narrative).await;
            db::update_character(
                pool, game_id,
                stats.hp, stats.xp, stats.gold, stats.level,
                &stats.inventory, &stats.location,
            ).await;
            broadcaster.send(game_id, GameEvent::DmNarration { text: narrative });
        }
        Err(e) => {
            broadcaster.send(game_id, GameEvent::Error { message: e });
        }
    }
}

struct ParsedStats {
    hp: i32,
    xp: i32,
    gold: i32,
    level: i32,
    inventory: String,
    location: String,
}

fn parse_response(response: &str, current: &db::CharacterRow) -> (String, ParsedStats) {
    let mut narrative = response.to_string();
    let mut stats = ParsedStats {
        hp: current.hp,
        xp: current.xp,
        gold: current.gold,
        level: current.level,
        inventory: current.inventory.clone(),
        location: current.location.clone(),
    };

    if let Some(json_start) = response.find("```json") {
        if let Some(json_end) = response[json_start..].find("```\n").or_else(|| {
            let after = &response[json_start + 7..];
            after.find("```").map(|i| i + 7)
        }) {
            let json_block = &response[json_start + 7..json_start + json_end];
            narrative = response[..json_start].trim().to_string();

            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_block.trim()) {
                if let Some(hp) = val.get("hp").and_then(|v| v.as_i64()) {
                    stats.hp = hp as i32;
                }
                if let Some(xp) = val.get("xp").and_then(|v| v.as_i64()) {
                    stats.xp = xp as i32;
                }
                if let Some(gold) = val.get("gold").and_then(|v| v.as_i64()) {
                    stats.gold = gold as i32;
                }
                if let Some(level) = val.get("level").and_then(|v| v.as_i64()) {
                    stats.level = level as i32;
                }
                if let Some(loc) = val.get("location").and_then(|v| v.as_str()) {
                    stats.location = loc.to_string();
                }
                if let Some(inv) = val.get("inventory").and_then(|v| v.as_array()) {
                    let items: Vec<String> = inv.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                    stats.inventory = serde_json::to_string(&items).unwrap_or_default();
                }
            }
        }
    }

    (narrative, stats)
}

pub async fn start_game(
    pool: &SqlitePool,
    broadcaster: &Arc<Broadcaster>,
    game_id: &str,
) {
    let game = db::get_game(pool, game_id).await.unwrap();
    let prompt = format!(
        r#"You are a Dungeon Master for a text-based RPG game. Create an opening scene for a new adventure.

Setting: {}
Player name: {}

Create a vivid, atmospheric opening (2-3 paragraphs) that:
- Describes where the player finds themselves
- Sets the mood and tone
- Hints at adventure or danger ahead
- Ends by asking what the player wants to do

IMPORTANT: At the end of your response, include a JSON block with the starting character stats:
```json
{{"hp": 100, "xp": 0, "gold": 10, "level": 1, "inventory": ["rusty sword", "torch", "bread"], "location": "<starting location name>"}}
```"#,
        game.setting,
        game.player_name,
    );

    broadcaster.send(game_id, GameEvent::DmThinking);

    match claude::call_claude(&prompt).await {
        Ok(response) => {
            let character = db::get_character(pool, game_id).await.unwrap();
            let (narrative, stats) = parse_response(&response, &character);
            db::save_message(pool, game_id, "dm", &narrative).await;
            db::update_character(
                pool, game_id,
                stats.hp, stats.xp, stats.gold, stats.level,
                &stats.inventory, &stats.location,
            ).await;
            broadcaster.send(game_id, GameEvent::DmNarration { text: narrative });
        }
        Err(e) => {
            broadcaster.send(game_id, GameEvent::Error { message: e });
        }
    }
}
