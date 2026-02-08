use std::sync::Arc;
use base64::Engine as _;
use base64::engine::general_purpose;
use sqlx::{Pool, Sqlite};
use crate::agents::get_runner;
use crate::persistence::db;
use crate::sse::broadcaster::{Broadcaster, BuildEvent};

pub struct BuildEngine {
    pool: Pool<Sqlite>,
    broadcaster: Arc<Broadcaster>,
}

impl BuildEngine {
    pub fn new(pool: Pool<Sqlite>, broadcaster: Arc<Broadcaster>) -> Self {
        Self { pool, broadcaster }
    }

    pub async fn run_build(&self, project_id: String, engine: String, image_data: String) {
        self.broadcaster.broadcast(&project_id, BuildEvent::StatusUpdate {
            step: "analyzing_drawing".to_string(),
            progress: 25,
        }).await;

        let output_dir = format!("output/{}", project_id);
        if let Err(e) = std::fs::create_dir_all(&output_dir) {
            self.broadcaster.broadcast(&project_id, BuildEvent::Error {
                message: format!("Failed to create output dir: {}", e),
            }).await;
            return;
        }

        let raw_b64 = if let Some(pos) = image_data.find(",") {
            &image_data[pos + 1..]
        } else {
            &image_data
        };

        let image_bytes = match general_purpose::STANDARD.decode(raw_b64) {
            Ok(bytes) => bytes,
            Err(e) => {
                self.broadcaster.broadcast(&project_id, BuildEvent::Error {
                    message: format!("Failed to decode image: {}", e),
                }).await;
                return;
            }
        };

        let image_path = format!("{}/drawing.png", output_dir);
        if let Err(e) = std::fs::write(&image_path, &image_bytes) {
            self.broadcaster.broadcast(&project_id, BuildEvent::Error {
                message: format!("Failed to write image: {}", e),
            }).await;
            return;
        }

        let abs_output_dir = std::fs::canonicalize(&output_dir)
            .unwrap_or_else(|_| std::path::PathBuf::from(&output_dir));
        let abs_image_path = std::fs::canonicalize(&image_path)
            .unwrap_or_else(|_| std::path::PathBuf::from(&image_path));

        let prompt = format!(
            r#"You are a web developer. I have a drawing of a website UI.
The drawing is saved at: {}

Look at the drawing and generate a complete, working website that matches it.

Requirements:
- Generate exactly 3 files: index.html, style.css, script.js
- The HTML must link to style.css and script.js
- The HTML must be semantic and responsive
- The CSS must be clean, modern, and match the layout in the drawing
- Add interactivity with vanilla JS where appropriate
- Do not use any frameworks or libraries
- Write all files to: {}

Output the files now."#,
            abs_image_path.display(),
            abs_output_dir.display()
        );

        self.broadcaster.broadcast(&project_id, BuildEvent::StatusUpdate {
            step: "generating_code".to_string(),
            progress: 50,
        }).await;

        let runner = get_runner(&engine);
        match runner.run(&prompt).await {
            Ok(response) => {
                self.broadcaster.broadcast(&project_id, BuildEvent::StatusUpdate {
                    step: "saving_files".to_string(),
                    progress: 75,
                }).await;

                ensure_output_files(&response, &output_dir);

                let _ = db::update_project_status(&self.pool, &project_id, "done", None).await;

                self.broadcaster.broadcast(&project_id, BuildEvent::StatusUpdate {
                    step: "done".to_string(),
                    progress: 100,
                }).await;

                self.broadcaster.broadcast(&project_id, BuildEvent::BuildComplete {
                    project_id: project_id.clone(),
                }).await;
            }
            Err(e) => {
                let _ = db::update_project_status(&self.pool, &project_id, "error", Some(&e)).await;

                self.broadcaster.broadcast(&project_id, BuildEvent::Error {
                    message: e,
                }).await;
            }
        }

        self.broadcaster.remove_channel(&project_id).await;
    }
}

fn ensure_output_files(response: &str, output_dir: &str) {
    let html_path = format!("{}/index.html", output_dir);
    let css_path = format!("{}/style.css", output_dir);
    let js_path = format!("{}/script.js", output_dir);

    if !std::path::Path::new(&html_path).exists() {
        let html = extract_block(response, "html").unwrap_or_else(|| {
            extract_block(response, "index.html").unwrap_or_else(|| {
                fallback_html()
            })
        });
        let _ = std::fs::write(&html_path, html);
    }

    if !std::path::Path::new(&css_path).exists() {
        let css = extract_block(response, "css").unwrap_or_else(|| {
            extract_block(response, "style.css").unwrap_or_default()
        });
        let _ = std::fs::write(&css_path, css);
    }

    if !std::path::Path::new(&js_path).exists() {
        let js = extract_block(response, "js").unwrap_or_else(|| {
            extract_block(response, "javascript").unwrap_or_else(|| {
                extract_block(response, "script.js").unwrap_or_default()
            })
        });
        let _ = std::fs::write(&js_path, js);
    }
}

fn extract_block(text: &str, lang: &str) -> Option<String> {
    let marker = format!("```{}", lang);
    if let Some(start) = text.find(&marker) {
        let after = &text[start + marker.len()..];
        let content_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
        let after = &after[content_start..];
        if let Some(end) = after.find("```") {
            return Some(after[..end].trim().to_string());
        }
    }
    None
}

fn fallback_html() -> String {
    r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Generated Site</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<h1>Site generation completed but no HTML was extracted</h1>
<script src="script.js"></script>
</body>
</html>"#.to_string()
}
