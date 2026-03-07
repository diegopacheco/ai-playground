use std::process::Stdio;
use tokio::process::Command;
use tokio::io::AsyncReadExt;

pub async fn apply_file(path: &str) -> Result<String, String> {
    let mut child = Command::new("kubectl")
        .args(&["apply", "-f", path])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to run kubectl: {}", e))?;

    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let mut out = String::new();
    let mut err = String::new();

    stdout.read_to_string(&mut out).await.map_err(|e| e.to_string())?;
    stderr.read_to_string(&mut err).await.map_err(|e| e.to_string())?;

    let status = child.wait().await.map_err(|e| e.to_string())?;

    if status.success() {
        Ok(format!("{}\n{}", out.trim(), err.trim()))
    } else {
        Err(format!("kubectl apply failed:\n{}\n{}", out, err))
    }
}
