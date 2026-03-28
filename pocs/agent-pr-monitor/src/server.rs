use actix_cors::Cors;
use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer};
use crate::state::SharedState;
use serde::Serialize;

#[derive(rust_embed::Embed)]
#[folder = "frontend/dist/"]
struct FrontendAssets;

#[derive(Serialize)]
struct StatusResponse {
    pr_info: crate::state::PrInfo,
    counters: crate::state::Counters,
    test_classification: crate::state::TestClassification,
}

#[derive(Serialize)]
struct HealthResponse {
    uptime_seconds: u64,
    total_cycles: u64,
    last_check: Option<String>,
}

#[derive(Serialize)]
struct FileContentResponse {
    path: String,
    content: String,
}

async fn api_status(state: web::Data<SharedState>) -> HttpResponse {
    let st = state.lock().unwrap();
    HttpResponse::Ok().json(StatusResponse {
        pr_info: st.pr_info.clone(),
        counters: st.counters.clone(),
        test_classification: st.test_classification.clone(),
    })
}

async fn api_actions(state: web::Data<SharedState>) -> HttpResponse {
    let st = state.lock().unwrap();
    HttpResponse::Ok().json(&st.actions)
}

async fn api_comments(state: web::Data<SharedState>) -> HttpResponse {
    let st = state.lock().unwrap();
    HttpResponse::Ok().json(&st.comments)
}

async fn api_files(state: web::Data<SharedState>) -> HttpResponse {
    let st = state.lock().unwrap();
    HttpResponse::Ok().json(&st.file_tree)
}

async fn api_file_content(
    state: web::Data<SharedState>,
    query: web::Query<std::collections::HashMap<String, String>>,
) -> HttpResponse {
    let path = match query.get("path") {
        Some(p) => p.clone(),
        None => return HttpResponse::BadRequest().body("Missing path parameter"),
    };
    let clone_path = {
        let st = state.lock().unwrap();
        st.pr_info.clone_path.clone()
    };
    let full_path = if path.starts_with('/') {
        path.clone()
    } else {
        format!("{}/{}", clone_path, path)
    };
    match crate::pr::read_file(&full_path) {
        Ok(content) => HttpResponse::Ok().json(FileContentResponse { path, content }),
        Err(e) => HttpResponse::NotFound().body(e),
    }
}

async fn api_logs(state: web::Data<SharedState>) -> HttpResponse {
    let st = state.lock().unwrap();
    HttpResponse::Ok().json(&st.logs)
}

async fn api_health(state: web::Data<SharedState>) -> HttpResponse {
    let st = state.lock().unwrap();
    let uptime = st.start_time.elapsed().as_secs();
    HttpResponse::Ok().json(HealthResponse {
        uptime_seconds: uptime,
        total_cycles: st.counters.total_cycles,
        last_check: st.last_check.clone(),
    })
}

async fn api_events(state: web::Data<SharedState>) -> HttpResponse {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    {
        let mut st = state.lock().unwrap();
        st.sse_clients.push(tx);
    }
    let stream = async_stream::stream! {
        while let Some(msg) = rx.recv().await {
            yield Ok::<_, actix_web::Error>(actix_web::web::Bytes::from(msg));
        }
    };
    HttpResponse::Ok()
        .insert_header(("Content-Type", "text/event-stream"))
        .insert_header(("Cache-Control", "no-cache"))
        .streaming(stream)
}

async fn api_conversation(state: web::Data<SharedState>) -> HttpResponse {
    let (owner, repo, pr_number) = {
        let st = state.lock().unwrap();
        (st.pr_info.owner.clone(), st.pr_info.repo.clone(), st.pr_info.pr_number)
    };
    let issue_comments = crate::pr::get_pr_review_comments(&owner, &repo, pr_number)
        .unwrap_or_else(|_| "[]".to_string());
    let review_comments = crate::pr::get_pr_comments(&owner, &repo, pr_number)
        .unwrap_or_else(|_| "[]".to_string());
    let issue_list: Vec<serde_json::Value> = serde_json::from_str(&issue_comments).unwrap_or_default();
    let review_list: Vec<serde_json::Value> = serde_json::from_str(&review_comments).unwrap_or_default();
    let mut all: Vec<serde_json::Value> = Vec::new();
    let pr_desc = crate::pr::get_pr_description(&owner, &repo, pr_number).unwrap_or_default();
    let pr_title = {
        let st = state.lock().unwrap();
        st.pr_info.title.clone()
    };
    if !pr_desc.is_empty() {
        all.push(serde_json::json!({
            "type": "description",
            "id": 0,
            "author": "PR Author",
            "body": format!("**{}**\n\n{}", pr_title, pr_desc),
            "created_at": "",
            "file_path": serde_json::Value::Null,
            "line": serde_json::Value::Null,
        }));
    }
    for c in issue_list {
        all.push(serde_json::json!({
            "type": "issue",
            "id": c["id"],
            "author": c["user"]["login"],
            "body": c["body"],
            "created_at": c["created_at"],
            "file_path": serde_json::Value::Null,
            "line": serde_json::Value::Null,
        }));
    }
    for c in review_list {
        all.push(serde_json::json!({
            "type": "review",
            "id": c["id"],
            "author": c["user"]["login"],
            "body": c["body"],
            "created_at": c["created_at"],
            "file_path": c["path"],
            "line": c["line"],
        }));
    }
    all.sort_by(|a, b| {
        let ta = a["created_at"].as_str().unwrap_or("");
        let tb = b["created_at"].as_str().unwrap_or("");
        ta.cmp(tb)
    });
    HttpResponse::Ok().json(all)
}

async fn api_trigger(state: web::Data<SharedState>) -> HttpResponse {
    let (clone_path, agent, model, owner, repo, pr_number, dry_run) = {
        let st = state.lock().unwrap();
        (
            st.pr_info.clone_path.clone(),
            st.pr_info.agent_name.clone(),
            st.pr_info.agent_model.clone(),
            st.pr_info.owner.clone(),
            st.pr_info.repo.clone(),
            st.pr_info.pr_number,
            st.dry_run,
        )
    };
    let inner_state: SharedState = state.get_ref().clone();
    tokio::task::spawn_blocking(move || {
        crate::monitor::run_single_cycle(
            &clone_path, &agent, &model, &owner, &repo, pr_number, dry_run, &inner_state,
        );
    });
    HttpResponse::Ok().json(serde_json::json!({"status": "triggered"}))
}

async fn serve_frontend(req: HttpRequest) -> HttpResponse {
    let path = req.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    match FrontendAssets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            HttpResponse::Ok()
                .content_type(mime.as_ref())
                .body(content.data.to_vec())
        }
        None => match FrontendAssets::get("index.html") {
            Some(content) => HttpResponse::Ok()
                .content_type("text/html")
                .body(content.data.to_vec()),
            None => HttpResponse::NotFound().body("Frontend not found"),
        },
    }
}

pub async fn start_server(state: SharedState, port: u16) {
    let state_data = web::Data::new(state);
    println!("Dashboard server starting on http://localhost:{}", port);
    HttpServer::new(move || {
        let cors = Cors::permissive();
        App::new()
            .wrap(cors)
            .app_data(state_data.clone())
            .route("/api/status", web::get().to(api_status))
            .route("/api/actions", web::get().to(api_actions))
            .route("/api/comments", web::get().to(api_comments))
            .route("/api/files", web::get().to(api_files))
            .route("/api/files/content", web::get().to(api_file_content))
            .route("/api/logs", web::get().to(api_logs))
            .route("/api/health", web::get().to(api_health))
            .route("/api/events", web::get().to(api_events))
            .route("/api/conversation", web::get().to(api_conversation))
            .route("/api/trigger", web::post().to(api_trigger))
            .default_service(web::get().to(serve_frontend))
    })
    .bind(("0.0.0.0", port))
    .expect("Failed to bind server")
    .run()
    .await
    .expect("Server failed");
}
