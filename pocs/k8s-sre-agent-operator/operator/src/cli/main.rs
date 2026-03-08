use std::env;
use std::process;
use std::process::Stdio;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use std::time::Duration;
use tokio::time::timeout;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let base_url = env::var("KOVALSKI_URL").unwrap_or_else(|_| "http://localhost:30080".to_string());

    let rt = tokio::runtime::Runtime::new().unwrap();
    match args[1].as_str() {
        "logs" => rt.block_on(do_get(&format!("{}/logs", base_url))),
        "fix" => rt.block_on(do_fix(&base_url)),
        "status" => rt.block_on(do_get(&format!("{}/status", base_url))),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
            process::exit(1);
        }
    }
}

async fn do_fix(base_url: &str) {
    println!("Collecting diagnostics from cluster...");
    let client = reqwest::Client::new();

    let diag = match client.get(&format!("{}/diagnostics", base_url)).send().await {
        Ok(resp) => {
            if !resp.status().is_success() {
                eprintln!("Failed to get diagnostics: {}", resp.text().await.unwrap_or_default());
                process::exit(1);
            }
            resp.text().await.unwrap_or_default()
        }
        Err(e) => {
            eprintln!("Failed to connect: {}", e);
            process::exit(1);
        }
    };

    if diag.trim().is_empty() {
        println!("No issues detected in the cluster.");
        return;
    }

    println!("Found issues. Asking Claude to fix...");
    println!("---");
    println!("{}", diag);
    println!("---");

    let prompt = format!(
        "You are a Kubernetes SRE expert. Analyze the following cluster diagnostics and produce ONLY the corrected YAML manifests. \
         Each YAML document must be separated by '---'. Do not include any explanation, markdown fences, or text outside the YAML. \
         Only output valid Kubernetes YAML that can be directly applied with kubectl apply -f.\n\n\
         DIAGNOSTICS:\n{}", diag
    );

    let claude_response = match run_claude(&prompt).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Claude error: {}", e);
            process::exit(1);
        }
    };

    let yaml_content = extract_yaml(&claude_response);

    if yaml_content.trim().is_empty() {
        eprintln!("Claude returned no valid YAML.");
        eprintln!("Raw response:\n{}", claude_response);
        process::exit(1);
    }

    println!("Applying fixes to cluster...");

    match client.post(&format!("{}/apply", base_url))
        .body(yaml_content.clone())
        .send()
        .await
    {
        Ok(resp) => {
            let body = resp.text().await.unwrap_or_default();
            println!("{}", body);
        }
        Err(e) => {
            eprintln!("Failed to apply: {}", e);
            process::exit(1);
        }
    }
}

async fn run_claude(prompt: &str) -> Result<String, String> {
    let mut child = Command::new("claude")
        .args(&["-p", prompt, "--dangerously-skip-permissions"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn claude: {}", e))?;

    let result = timeout(Duration::from_secs(180), async {
        let mut stdout = child.stdout.take().unwrap();
        let mut output = String::new();
        stdout.read_to_string(&mut output).await.map_err(|e| e.to_string())?;
        child.wait().await.map_err(|e| e.to_string())?;
        Ok::<String, String>(output)
    })
    .await;

    match result {
        Ok(Ok(output)) => {
            let trimmed = output.trim().to_string();
            if trimmed.is_empty() {
                Err("Claude returned empty response".to_string())
            } else {
                Ok(trimmed)
            }
        }
        Ok(Err(e)) => Err(e),
        Err(_) => {
            let _ = child.kill().await;
            Err("Claude timed out after 180s".to_string())
        }
    }
}

fn extract_yaml(response: &str) -> String {
    let mut blocks: Vec<String> = Vec::new();
    let mut current_block: Vec<&str> = Vec::new();
    let mut in_fence = false;

    for line in response.lines() {
        if line.starts_with("```") {
            if in_fence && !current_block.is_empty() {
                blocks.push(current_block.join("\n"));
                current_block.clear();
            }
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            current_block.push(line);
        }
    }

    if !blocks.is_empty() {
        return blocks.join("\n---\n");
    }

    response.to_string()
}

async fn do_get(url: &str) {
    match reqwest::get(url).await {
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status.is_success() {
                println!("{}", body);
            } else {
                eprintln!("Error ({}): {}", status, body);
                process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Failed to connect: {}", e);
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("Usage: kovalski <command>");
    eprintln!("");
    eprintln!("Commands:");
    eprintln!("  logs    Read all pod logs from the cluster");
    eprintln!("  fix     Fix broken deployments using Claude AI");
    eprintln!("  status  Show all resources in the cluster (kubectl get all)");
    eprintln!("");
    eprintln!("Environment:");
    eprintln!("  KOVALSKI_URL  Base URL of the SRE agent (default: http://localhost:30080)");
}
