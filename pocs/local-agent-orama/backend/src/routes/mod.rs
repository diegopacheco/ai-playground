use actix_web::{web, HttpResponse, Responder};
use crate::models::{RunRequest, RunResponse, StatusResponse, AgentInfo, FilesResponse};
use crate::agents::{self, SharedAgentState, create_agent_state};
use crate::worktree::{create_workspace, create_agent_worktree};
use crate::files::{list_files, read_file};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

pub type SessionStore = Arc<RwLock<HashMap<String, Vec<SharedAgentState>>>>;

pub fn create_session_store() -> SessionStore {
    Arc::new(RwLock::new(HashMap::new()))
}

#[derive(Clone)]
pub struct AgentConfig {
    pub key: &'static str,
    pub name: &'static str,
    pub model: &'static str,
}

const AGENTS: [AgentConfig; 4] = [
    AgentConfig { key: "claude-code", name: "Claude Code", model: "claude-sonnet-4-20250514" },
    AgentConfig { key: "codex", name: "Codex", model: "o4-mini" },
    AgentConfig { key: "gemini", name: "Gemini", model: "gemini-2.5-pro" },
    AgentConfig { key: "copilot", name: "Copilot", model: "gpt-4o" },
];

pub async fn run_agents(
    req: web::Json<RunRequest>,
    store: web::Data<SessionStore>,
) -> impl Responder {
    let session_id = Uuid::new_v4().to_string();
    let base_path = match create_workspace(&req.project_name).await {
        Ok(p) => p,
        Err(e) => return HttpResponse::InternalServerError().body(e),
    };
    let mut agent_states = Vec::new();
    for config in &AGENTS {
        let worktree = match create_agent_worktree(&base_path, config.key).await {
            Ok(p) => p,
            Err(e) => return HttpResponse::InternalServerError().body(e),
        };
        let state = create_agent_state(config.name, config.model, worktree);
        agent_states.push(state);
    }
    {
        let mut sessions = store.write().await;
        sessions.insert(session_id.clone(), agent_states.clone());
    }
    let prompt = req.prompt.clone();
    for (i, state) in agent_states.iter().enumerate() {
        let s = state.clone();
        let p = prompt.clone();
        match i {
            0 => tokio::spawn(async move { agents::claude::run(s, &p).await }),
            1 => tokio::spawn(async move { agents::codex::run(s, &p).await }),
            2 => tokio::spawn(async move { agents::gemini::run(s, &p).await }),
            3 => tokio::spawn(async move { agents::copilot::run(s, &p).await }),
            _ => unreachable!(),
        };
    }
    HttpResponse::Ok().json(RunResponse { session_id })
}

pub async fn get_status(
    path: web::Path<String>,
    store: web::Data<SessionStore>,
) -> impl Responder {
    let session_id = path.into_inner();
    let sessions = store.read().await;
    let states = match sessions.get(&session_id) {
        Some(s) => s,
        None => return HttpResponse::NotFound().body("Session not found"),
    };
    let mut agents = Vec::new();
    for state in states {
        let s = state.lock().await;
        agents.push(AgentInfo {
            name: s.name.clone(),
            model: s.model.clone(),
            status: s.status,
            worktree: s.worktree.to_string_lossy().to_string(),
        });
    }
    HttpResponse::Ok().json(StatusResponse { agents })
}

pub async fn get_files(
    path: web::Path<(String, String)>,
    store: web::Data<SessionStore>,
) -> impl Responder {
    let (session_id, agent_name) = path.into_inner();
    let sessions = store.read().await;
    let states = match sessions.get(&session_id) {
        Some(s) => s,
        None => return HttpResponse::NotFound().body("Session not found"),
    };
    let agent_key = agent_name.to_lowercase().replace(' ', "-");
    for state in states {
        let s = state.lock().await;
        let state_key = s.name.to_lowercase().replace(' ', "-");
        if state_key == agent_key {
            let files = list_files(&s.worktree);
            return HttpResponse::Ok().json(FilesResponse { files });
        }
    }
    HttpResponse::NotFound().body("Agent not found")
}

#[derive(serde::Deserialize)]
pub struct FileQuery {
    path: String,
}

pub async fn get_file_content(
    path: web::Path<(String, String)>,
    query: web::Query<FileQuery>,
    store: web::Data<SessionStore>,
) -> impl Responder {
    let (session_id, agent_name) = path.into_inner();
    let sessions = store.read().await;
    let states = match sessions.get(&session_id) {
        Some(s) => s,
        None => return HttpResponse::NotFound().body("Session not found"),
    };
    let agent_key = agent_name.to_lowercase().replace(' ', "-");
    for state in states {
        let s = state.lock().await;
        let state_key = s.name.to_lowercase().replace(' ', "-");
        if state_key == agent_key {
            return match read_file(&s.worktree, &query.path).await {
                Ok(content) => HttpResponse::Ok().json(content),
                Err(e) => HttpResponse::NotFound().body(e),
            };
        }
    }
    HttpResponse::NotFound().body("Agent not found")
}
