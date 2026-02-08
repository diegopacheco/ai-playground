use actix_web::{web, HttpResponse};
use base64::Engine as _;
use base64::engine::general_purpose;
use chrono::Utc;
use uuid::Uuid;
use crate::AppState;
use crate::agents::get_runner;
use crate::persistence::db;
use crate::persistence::models::{CreateGuessRequest, CreateGuessResponse, GuessRecord};

pub async fn create_guess(
    state: web::Data<AppState>,
    req: web::Json<CreateGuessRequest>,
) -> HttpResponse {
    let id = Uuid::new_v4().to_string();

    let output_dir = format!("output/{}", id);
    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        return HttpResponse::InternalServerError().json(CreateGuessResponse {
            id: id.clone(),
            guess: None,
            engine: req.engine.clone(),
            status: "error".to_string(),
            error: Some(format!("Failed to create output dir: {}", e)),
        });
    }

    let raw_b64 = if let Some(pos) = req.image.find(",") {
        &req.image[pos + 1..]
    } else {
        &req.image
    };

    let image_bytes = match general_purpose::STANDARD.decode(raw_b64) {
        Ok(bytes) => bytes,
        Err(e) => {
            return HttpResponse::BadRequest().json(CreateGuessResponse {
                id: id.clone(),
                guess: None,
                engine: req.engine.clone(),
                status: "error".to_string(),
                error: Some(format!("Failed to decode image: {}", e)),
            });
        }
    };

    let image_path = format!("{}/drawing.png", output_dir);
    if let Err(e) = std::fs::write(&image_path, &image_bytes) {
        return HttpResponse::InternalServerError().json(CreateGuessResponse {
            id: id.clone(),
            guess: None,
            engine: req.engine.clone(),
            status: "error".to_string(),
            error: Some(format!("Failed to write image: {}", e)),
        });
    }

    let record = GuessRecord {
        id: id.clone(),
        engine: req.engine.clone(),
        guess: None,
        status: "pending".to_string(),
        created_at: Utc::now().to_rfc3339(),
        completed_at: None,
        error: None,
    };
    let _ = db::create_guess(&state.pool, &record).await;

    let abs_image_path = std::fs::canonicalize(&image_path)
        .unwrap_or_else(|_| std::path::PathBuf::from(&image_path));

    let prompt = format!(
        "Look at the image at: {}\nWhat object is drawn in this image? Your ENTIRE response must be exactly 1 or 2 words. No explanation, no sentences, no punctuation. Just the name of the object. For example: Cat or Red Car",
        abs_image_path.display()
    );

    let runner = get_runner(&req.engine);
    match runner.run(&prompt).await {
        Ok(response) => {
            let guess = extract_guess(&response);
            let _ = db::update_guess(&state.pool, &id, Some(&guess), "done", None).await;
            HttpResponse::Ok().json(CreateGuessResponse {
                id,
                guess: Some(guess),
                engine: req.engine.clone(),
                status: "done".to_string(),
                error: None,
            })
        }
        Err(e) => {
            let _ = db::update_guess(&state.pool, &id, None, "error", Some(&e)).await;
            HttpResponse::Ok().json(CreateGuessResponse {
                id,
                guess: None,
                engine: req.engine.clone(),
                status: "error".to_string(),
                error: Some(e),
            })
        }
    }
}

fn extract_guess(response: &str) -> String {
    let last_line = response
        .trim()
        .lines()
        .rev()
        .find(|line| {
            let l = line.trim();
            !l.is_empty()
                && !l.starts_with('#')
                && !l.starts_with('*')
                && !l.starts_with('`')
                && !l.starts_with('-')
                && !l.starts_with('>')
                && !l.contains("tool")
                && !l.contains("Read")
                && !l.contains("file")
                && !l.contains("image")
                && !l.contains("drawing")
                && !l.contains("saved at")
                && l.split_whitespace().count() <= 5
        })
        .unwrap_or_else(|| {
            response.trim().lines().last().unwrap_or("Unknown")
        });

    let words: Vec<&str> = last_line
        .trim()
        .trim_matches(|c: char| c == '.' || c == '!' || c == '"' || c == '\'' || c == '*' || c == '`')
        .split_whitespace()
        .collect();

    if words.is_empty() {
        return "Unknown".to_string();
    }
    words.into_iter().take(2).collect::<Vec<&str>>().join(" ")
}

pub async fn get_guesses(state: web::Data<AppState>) -> HttpResponse {
    let records = db::get_all_guesses(&state.pool).await.unwrap_or_default();
    HttpResponse::Ok().json(records)
}

pub async fn get_guess(
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> HttpResponse {
    let id = path.into_inner();
    let record = db::get_guess(&state.pool, &id).await.ok().flatten();
    HttpResponse::Ok().json(record)
}

pub async fn delete_guess(
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> HttpResponse {
    let id = path.into_inner();
    let output_dir = format!("output/{}", id);
    let _ = std::fs::remove_dir_all(&output_dir);
    let result = db::delete_guess(&state.pool, &id).await.is_ok();
    HttpResponse::Ok().json(result)
}
