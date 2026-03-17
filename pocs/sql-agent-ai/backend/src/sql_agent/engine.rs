use crate::agents::claude;
use crate::app_state::{AppState, QueryEvent};
use crate::persistence::db;
use sqlx::{Column, Row};
use std::sync::Arc;

const MAX_RETRIES: u32 = 3;

const BLOCKED_KEYWORDS: &[&str] = &[
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
];

fn is_dangerous_sql(sql: &str) -> bool {
    let upper = sql.to_uppercase();
    BLOCKED_KEYWORDS.iter().any(|kw| {
        let pattern = format!(r"\b{}\b", kw);
        regex_lite_match(&upper, &pattern)
    })
}

fn regex_lite_match(text: &str, keyword: &str) -> bool {
    text.split_whitespace().any(|word| {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        clean == keyword.trim_start_matches(r"\b").trim_end_matches(r"\b")
    })
}

fn build_sql_prompt(question: &str, schema: &str) -> String {
    format!(
        "You are a SQL expert. Convert the following natural language question into a PostgreSQL SELECT query.\n\
        \n\
        DATABASE SCHEMA:\n{}\n\
        \n\
        RULES:\n\
        - Only generate SELECT queries, never INSERT, UPDATE, DELETE, DROP, or any data modification.\n\
        - Return ONLY the SQL query, nothing else. No markdown, no explanation, no code fences.\n\
        - Use proper JOINs when needed.\n\
        - Use meaningful column aliases.\n\
        - For forecasting, use simple linear extrapolation based on existing data trends.\n\
        \n\
        QUESTION: {}",
        schema, question
    )
}

fn build_fix_prompt(question: &str, schema: &str, sql: &str, error: &str) -> String {
    format!(
        "You are a SQL expert. The following PostgreSQL query failed. Fix it.\n\
        \n\
        DATABASE SCHEMA:\n{}\n\
        \n\
        ORIGINAL QUESTION: {}\n\
        \n\
        FAILED SQL:\n{}\n\
        \n\
        ERROR:\n{}\n\
        \n\
        RULES:\n\
        - Only generate SELECT queries.\n\
        - Return ONLY the fixed SQL query, nothing else. No markdown, no explanation, no code fences.\n\
        - Make sure column names and table names match the schema exactly.",
        schema, question, sql, error
    )
}

fn extract_sql(response: &str) -> String {
    let trimmed = response.trim();
    if trimmed.starts_with("```") {
        let lines: Vec<&str> = trimmed.lines().collect();
        let start = if lines.first().map_or(false, |l| l.starts_with("```")) { 1 } else { 0 };
        let end = if lines.last().map_or(false, |l| l.starts_with("```")) { lines.len() - 1 } else { lines.len() };
        lines[start..end].join("\n").trim().to_string()
    } else {
        trimmed.to_string()
    }
}

pub async fn run_query(state: Arc<AppState>, query_id: String, question: String) {
    let schema = get_db_schema(&state).await;

    state.broadcaster.send(&query_id, QueryEvent::Thinking {
        message: "Converting your question to SQL...".to_string(),
    }).await;

    let prompt = build_sql_prompt(&question, &schema);
    let llm_response = match claude::call_claude(&prompt).await {
        Ok(r) => r,
        Err(e) => {
            state.broadcaster.send(&query_id, QueryEvent::Failed {
                error: format!("LLM call failed: {}", e),
            }).await;
            state.broadcaster.remove(&query_id).await;
            return;
        }
    };

    let mut sql = extract_sql(&llm_response);
    let mut attempt = 1u32;

    if is_dangerous_sql(&sql) {
        state.broadcaster.send(&query_id, QueryEvent::Failed {
            error: "Blocked: Only SELECT queries are allowed. Data modification is not permitted.".to_string(),
        }).await;
        db::update_query_result(&state.pool, &query_id, "blocked", &sql, "Only SELECT queries allowed").await;
        state.broadcaster.remove(&query_id).await;
        return;
    }

    state.broadcaster.send(&query_id, QueryEvent::SqlGenerated {
        sql: sql.clone(),
        attempt,
    }).await;

    loop {
        match execute_sql(&state, &sql).await {
            Ok((columns, rows)) => {
                state.broadcaster.send(&query_id, QueryEvent::QueryResult {
                    columns: columns.clone(),
                    rows: rows.clone(),
                    sql: sql.clone(),
                }).await;
                let result_json = serde_json::json!({ "columns": columns, "rows": rows });
                db::update_query_result(&state.pool, &query_id, "success", &sql, &result_json.to_string()).await;
                break;
            }
            Err(error) => {
                state.broadcaster.send(&query_id, QueryEvent::SqlError {
                    error: error.clone(),
                    attempt,
                }).await;

                if attempt >= MAX_RETRIES {
                    state.broadcaster.send(&query_id, QueryEvent::Failed {
                        error: format!("Failed after {} attempts. Last error: {}", MAX_RETRIES, error),
                    }).await;
                    db::update_query_result(&state.pool, &query_id, "failed", &sql, &error).await;
                    break;
                }

                attempt += 1;
                state.broadcaster.send(&query_id, QueryEvent::Thinking {
                    message: format!("Fixing SQL (attempt {}/{})", attempt, MAX_RETRIES),
                }).await;

                let fix_prompt = build_fix_prompt(&question, &schema, &sql, &error);
                match claude::call_claude(&fix_prompt).await {
                    Ok(fixed) => {
                        sql = extract_sql(&fixed);
                        if is_dangerous_sql(&sql) {
                            state.broadcaster.send(&query_id, QueryEvent::Failed {
                                error: "Blocked: Fixed query contained data modification statements.".to_string(),
                            }).await;
                            db::update_query_result(&state.pool, &query_id, "blocked", &sql, "Only SELECT queries allowed").await;
                            break;
                        }
                        state.broadcaster.send(&query_id, QueryEvent::SqlFixed {
                            sql: sql.clone(),
                            attempt,
                        }).await;
                    }
                    Err(e) => {
                        state.broadcaster.send(&query_id, QueryEvent::Failed {
                            error: format!("LLM fix call failed: {}", e),
                        }).await;
                        db::update_query_result(&state.pool, &query_id, "failed", &sql, &e).await;
                        break;
                    }
                }
            }
        }
    }

    state.broadcaster.remove(&query_id).await;
}

async fn execute_sql(state: &AppState, sql: &str) -> Result<(Vec<String>, Vec<Vec<serde_json::Value>>), String> {
    let rows = sqlx::query(sql)
        .fetch_all(&state.pool)
        .await
        .map_err(|e| e.to_string())?;

    if rows.is_empty() {
        return Ok((vec![], vec![]));
    }

    let columns: Vec<String> = rows[0]
        .columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();

    let mut result_rows = Vec::new();
    for row in &rows {
        let mut row_values = Vec::new();
        for col in &columns {
            let value: serde_json::Value = try_extract_value(row, col);
            row_values.push(value);
        }
        result_rows.push(row_values);
    }

    Ok((columns, result_rows))
}

fn try_extract_value(row: &sqlx::postgres::PgRow, col: &str) -> serde_json::Value {
    if let Ok(v) = row.try_get::<i64, _>(col.as_ref() as &str) {
        return serde_json::Value::Number(v.into());
    }
    if let Ok(v) = row.try_get::<i32, _>(col.as_ref() as &str) {
        return serde_json::Value::Number(v.into());
    }
    if let Ok(v) = row.try_get::<f64, _>(col.as_ref() as &str) {
        return serde_json::json!(v);
    }
    if let Ok(v) = row.try_get::<bigdecimal::BigDecimal, _>(col.as_ref() as &str) {
        let s = v.to_string();
        if let Ok(f) = s.parse::<f64>() {
            return serde_json::json!(f);
        }
        return serde_json::Value::String(s);
    }
    if let Ok(v) = row.try_get::<chrono::NaiveDate, _>(col.as_ref() as &str) {
        return serde_json::Value::String(v.to_string());
    }
    if let Ok(v) = row.try_get::<chrono::NaiveDateTime, _>(col.as_ref() as &str) {
        return serde_json::Value::String(v.to_string());
    }
    if let Ok(v) = row.try_get::<String, _>(col.as_ref() as &str) {
        return serde_json::Value::String(v);
    }
    if let Ok(v) = row.try_get::<bool, _>(col.as_ref() as &str) {
        return serde_json::Value::Bool(v);
    }
    serde_json::Value::Null
}

async fn get_db_schema(state: &AppState) -> String {
    let tables = sqlx::query(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
    )
    .fetch_all(&state.pool)
    .await
    .unwrap_or_default();

    let mut schema = String::new();
    for table_row in &tables {
        let table_name: String = table_row.get("table_name");
        schema.push_str(&format!("TABLE: {}\n", table_name));

        let cols = sqlx::query(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema = 'public' AND table_name = $1 ORDER BY ordinal_position"
        )
        .bind(&table_name)
        .fetch_all(&state.pool)
        .await
        .unwrap_or_default();

        for col in &cols {
            let col_name: String = col.get("column_name");
            let data_type: String = col.get("data_type");
            let nullable: String = col.get("is_nullable");
            schema.push_str(&format!("  - {} ({}, nullable: {})\n", col_name, data_type, nullable));
        }
        schema.push('\n');
    }
    schema
}
