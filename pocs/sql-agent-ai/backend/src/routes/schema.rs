use axum::extract::State;
use axum::Json;
use sqlx::Row;
use std::sync::Arc;

use crate::app_state::AppState;

#[derive(serde::Serialize)]
pub struct TableInfo {
    pub name: String,
    pub columns: Vec<ColumnInfo>,
}

#[derive(serde::Serialize)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: String,
}

pub async fn get_schema(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<TableInfo>> {
    let tables = sqlx::query(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name"
    )
    .fetch_all(&state.pool)
    .await
    .unwrap_or_default();

    let mut result = Vec::new();
    for table_row in &tables {
        let table_name: String = table_row.get("table_name");
        let cols = sqlx::query(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = $1 ORDER BY ordinal_position"
        )
        .bind(&table_name)
        .fetch_all(&state.pool)
        .await
        .unwrap_or_default();

        let columns: Vec<ColumnInfo> = cols.iter().map(|c| ColumnInfo {
            name: c.get("column_name"),
            data_type: c.get("data_type"),
        }).collect();

        result.push(TableInfo {
            name: table_name,
            columns,
        });
    }

    Json(result)
}
