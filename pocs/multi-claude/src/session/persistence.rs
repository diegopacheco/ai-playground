use std::fs;
use std::path::PathBuf;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use crate::agent::AgentType;

#[derive(Serialize, Deserialize)]
pub struct LayoutData {
    pub version: u32,
    pub last_saved: DateTime<Utc>,
    pub sessions: Vec<SessionData>,
    pub active_session_index: Option<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct SessionData {
    pub id: String,
    pub agent_type: AgentType,
    pub working_dir: PathBuf,
}

fn layout_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join("multi-claude").join("sessions").join("layout.json")
}

pub fn save_layout(
    sessions: &[(&str, AgentType, &PathBuf)],
    active_index: Option<usize>,
) -> Result<()> {
    let path = layout_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let layout = LayoutData {
        version: 1,
        last_saved: Utc::now(),
        sessions: sessions
            .iter()
            .map(|(id, agent_type, working_dir)| SessionData {
                id: id.to_string(),
                agent_type: *agent_type,
                working_dir: (*working_dir).clone(),
            })
            .collect(),
        active_session_index: active_index,
    };

    let json = serde_json::to_string_pretty(&layout)?;
    fs::write(&path, json)?;
    Ok(())
}

pub fn load_layout() -> Result<Option<LayoutData>> {
    let path = layout_path();
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(&path)?;
    let layout: LayoutData = serde_json::from_str(&content)?;
    Ok(Some(layout))
}
