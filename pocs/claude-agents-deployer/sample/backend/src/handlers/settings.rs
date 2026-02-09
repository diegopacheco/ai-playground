use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use sqlx::PgPool;

use crate::models::settings::{AppSettings, UpdateSettings};

fn is_valid_theme(theme: &str) -> bool {
    matches!(theme, "classic" | "forest" | "sunset")
}

pub async fn get_settings(State(pool): State<PgPool>) -> impl IntoResponse {
    let ensure = sqlx::query(
        "INSERT INTO settings (id, comments_enabled, background_theme, updated_at)
         VALUES (1, TRUE, 'classic', NOW())
         ON CONFLICT (id) DO NOTHING",
    )
    .execute(&pool)
    .await;

    if let Err(e) = ensure {
        tracing::error!("Failed to ensure settings row: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response();
    }

    let settings = sqlx::query_as::<_, AppSettings>("SELECT * FROM settings WHERE id = 1")
        .fetch_one(&pool)
        .await;

    match settings {
        Ok(settings) => (StatusCode::OK, Json(serde_json::json!(settings))).into_response(),
        Err(e) => {
            tracing::error!("Failed to fetch settings: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

pub async fn update_settings(
    State(pool): State<PgPool>,
    Json(payload): Json<UpdateSettings>,
) -> impl IntoResponse {
    if let Some(theme) = payload.background_theme.as_deref() {
        if !is_valid_theme(theme) {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid background theme"})),
            )
                .into_response();
        }
    }

    let ensure = sqlx::query(
        "INSERT INTO settings (id, comments_enabled, background_theme, updated_at)
         VALUES (1, TRUE, 'classic', NOW())
         ON CONFLICT (id) DO NOTHING",
    )
    .execute(&pool)
    .await;

    if let Err(e) = ensure {
        tracing::error!("Failed to ensure settings row: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response();
    }

    let updated = sqlx::query_as::<_, AppSettings>(
        "UPDATE settings
         SET comments_enabled = COALESCE($1, comments_enabled),
             background_theme = COALESCE($2, background_theme),
             updated_at = NOW()
         WHERE id = 1
         RETURNING *",
    )
    .bind(payload.comments_enabled)
    .bind(payload.background_theme.as_deref())
    .fetch_one(&pool)
    .await;

    match updated {
        Ok(settings) => (StatusCode::OK, Json(serde_json::json!(settings))).into_response(),
        Err(e) => {
            tracing::error!("Failed to update settings: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}
