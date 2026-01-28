use actix_web::{web, HttpResponse, Responder};
use std::fs;
use std::sync::Arc;
use uuid::Uuid;
use crate::agents::{get_default_model, is_valid_agent};
use crate::cycle::{get_solutions_dir, run_learning_cycles_with_callback, ProgressCallback};
use super::models::*;
use super::state::AppState;

pub async fn list_projects(state: web::Data<Arc<AppState>>) -> impl Responder {
    let solutions_dir = state.base_dir.join(get_solutions_dir());
    let mut projects = Vec::new();
    if let Ok(entries) = fs::read_dir(&solutions_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                let project_path = entry.path();
                let mut cycles = Vec::new();
                if let Ok(sub_entries) = fs::read_dir(&project_path) {
                    for sub in sub_entries.flatten() {
                        let sub_name = sub.file_name().to_string_lossy().to_string();
                        if sub_name.starts_with("cycle-") {
                            cycles.push(sub_name);
                        }
                    }
                }
                cycles.sort();
                let has_memory = project_path.join("memory.txt").exists();
                let has_mistakes = project_path.join("mistakes.txt").exists();
                projects.push(ProjectInfo { name, cycles, has_memory, has_mistakes });
            }
        }
    }
    projects.sort_by(|a, b| a.name.cmp(&b.name));
    HttpResponse::Ok().json(projects)
}

pub async fn get_project(
    path: web::Path<String>,
    state: web::Data<Arc<AppState>>,
) -> impl Responder {
    let project_name = path.into_inner();
    let project_path = state.base_dir.join(get_solutions_dir()).join(&project_name);
    if !project_path.exists() {
        return HttpResponse::NotFound().json(serde_json::json!({"error": "Project not found"}));
    }
    let memory = fs::read_to_string(project_path.join("memory.txt")).unwrap_or_default();
    let mistakes = fs::read_to_string(project_path.join("mistakes.txt")).unwrap_or_default();
    let prompts = fs::read_to_string(project_path.join("prompts.md")).unwrap_or_default();
    let mut cycles = Vec::new();
    if let Ok(entries) = fs::read_dir(&project_path) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("cycle-") {
                if let Some(num_str) = name.strip_prefix("cycle-") {
                    if let Ok(num) = num_str.parse::<u32>() {
                        let cycle_path = entry.path();
                        let learnings_path = cycle_path.join("learnings.txt");
                        let mistakes_path = cycle_path.join("mistakes.txt");
                        let improved_prompt_path = cycle_path.join("improved_prompt.txt");
                        cycles.push(CycleInfo {
                            cycle_number: num,
                            has_prompt: cycle_path.join("prompt.txt").exists(),
                            has_output: cycle_path.join("output.txt").exists(),
                            has_review: cycle_path.join("review.txt").exists(),
                            has_learnings: learnings_path.exists(),
                            has_mistakes: mistakes_path.exists(),
                            has_improved_prompt: improved_prompt_path.exists(),
                            learnings_content: fs::read_to_string(&learnings_path).unwrap_or_default(),
                            mistakes_content: fs::read_to_string(&mistakes_path).unwrap_or_default(),
                            improved_prompt_content: fs::read_to_string(&improved_prompt_path).unwrap_or_default(),
                        });
                    }
                }
            }
        }
    }
    cycles.sort_by_key(|c| c.cycle_number);
    HttpResponse::Ok().json(ProjectDetail {
        name: project_name,
        memory,
        mistakes,
        prompts,
        cycles,
    })
}

pub async fn submit_task(
    body: web::Json<TaskRequest>,
    state: web::Data<Arc<AppState>>,
) -> impl Responder {
    let task_id = Uuid::new_v4().to_string();
    let cfg = state.get_config().await;
    let agent = body.agent.clone().unwrap_or(cfg.agent.clone());
    let model = body.model.clone().unwrap_or(cfg.model.clone());
    let cycles = body.cycles.unwrap_or(cfg.cycles);
    if !is_valid_agent(&agent) {
        return HttpResponse::BadRequest().json(serde_json::json!({"error": "Invalid agent"}));
    }
    let model = if model.is_empty() {
        get_default_model(&agent).to_string()
    } else {
        model
    };
    let status = TaskStatus {
        task_id: task_id.clone(),
        status: "pending".to_string(),
        current_cycle: 0,
        total_cycles: cycles,
        phase: "starting".to_string(),
        completed: false,
        success: false,
    };
    state.add_task(task_id.clone(), status).await;
    let task_clone = body.task.clone();
    let base_dir = state.base_dir.clone();
    let state_clone = state.clone();
    let tid = task_id.clone();
    tokio::spawn(async move {
        state_clone.update_task(&tid, |t| {
            t.status = "running".to_string();
        }).await;
        state_clone.send_event(ProgressEvent {
            task_id: tid.clone(),
            event_type: "start".to_string(),
            cycle: 0,
            phase: "starting".to_string(),
            message: "Task started".to_string(),
        });
        let tid_cb = tid.clone();
        let state_cb = state_clone.clone();
        let progress_cb: ProgressCallback = std::sync::Arc::new(move |cycle, phase, msg| {
            state_cb.update_task_sync(&tid_cb, |t| {
                t.current_cycle = cycle;
                t.phase = phase.to_string();
            });
            state_cb.send_event(ProgressEvent {
                task_id: tid_cb.clone(),
                event_type: "progress".to_string(),
                cycle,
                phase: phase.to_string(),
                message: msg.to_string(),
            });
        });
        let reports = run_learning_cycles_with_callback(&base_dir, &task_clone, &agent, &model, cycles, Some(progress_cb)).await;
        let success = reports.iter().any(|r| r.success);
        state_clone.update_task(&tid, |t| {
            t.status = if success { "completed" } else { "failed" }.to_string();
            t.completed = true;
            t.success = success;
            t.current_cycle = cycles;
            t.phase = "done".to_string();
        }).await;
        state_clone.send_event(ProgressEvent {
            task_id: tid.clone(),
            event_type: "complete".to_string(),
            cycle: cycles,
            phase: "done".to_string(),
            message: if success { "Task completed successfully" } else { "Task failed" }.to_string(),
        });
    });
    HttpResponse::Accepted().json(TaskResponse {
        task_id,
        status: "pending".to_string(),
    })
}

pub async fn get_task_status(
    path: web::Path<String>,
    state: web::Data<Arc<AppState>>,
) -> impl Responder {
    let task_id = path.into_inner();
    match state.get_task(&task_id).await {
        Some(status) => HttpResponse::Ok().json(status),
        None => HttpResponse::NotFound().json(serde_json::json!({"error": "Task not found"})),
    }
}

pub async fn get_config(state: web::Data<Arc<AppState>>) -> impl Responder {
    HttpResponse::Ok().json(state.get_config().await)
}

pub async fn update_config(
    body: web::Json<ConfigRequest>,
    state: web::Data<Arc<AppState>>,
) -> impl Responder {
    if let Some(ref agent) = body.agent {
        if !is_valid_agent(agent) {
            return HttpResponse::BadRequest().json(serde_json::json!({"error": "Invalid agent"}));
        }
    }
    if let Some(cycles) = body.cycles {
        if cycles == 0 || cycles > 10 {
            return HttpResponse::BadRequest().json(serde_json::json!({"error": "Cycles must be between 1 and 10"}));
        }
    }
    state.update_config(body.agent.clone(), body.model.clone(), body.cycles).await;
    HttpResponse::Ok().json(state.get_config().await)
}

pub async fn health() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({"status": "ok"}))
}
