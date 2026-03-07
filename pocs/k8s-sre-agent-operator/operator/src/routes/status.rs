use axum::http::StatusCode;
use std::process::Stdio;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

pub async fn get_status() -> Result<String, (StatusCode, String)> {
    let mut child = Command::new("kubectl")
        .args(&["get", "all", "-A"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut stdout = child.stdout.take().unwrap();
    let mut output = String::new();
    stdout.read_to_string(&mut output).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    child.wait().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(output)
}
