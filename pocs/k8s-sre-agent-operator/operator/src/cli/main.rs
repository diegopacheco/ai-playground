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
        "logs-summary" => rt.block_on(do_logs_summary(&base_url)),
        "ui" => open_ui(&base_url),
        "deploy" => rt.block_on(do_deploy()),
        "k8s" => rt.block_on(do_k8s()),
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

    let project_dir = find_project_dir();
    let fixed_dir = format!("{}/fixed-specs", project_dir);
    std::fs::create_dir_all(&fixed_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create fixed-specs dir: {}", e);
        process::exit(1);
    });

    let docs: Vec<&str> = yaml_content.split("\n---\n").collect();
    for (i, doc) in docs.iter().enumerate() {
        let name = extract_name(doc).unwrap_or_else(|| format!("fix-{}", i));
        let path = format!("{}/{}.yaml", fixed_dir, name);
        std::fs::write(&path, doc.trim()).unwrap_or_else(|e| {
            eprintln!("Failed to write {}: {}", path, e);
        });
        println!("Saved fixed spec: {}", path);
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

fn find_project_dir() -> String {
    let exe = env::current_exe().unwrap_or_default();
    let mut dir = exe.parent().unwrap().to_path_buf();
    loop {
        if dir.join("specs").exists() {
            return dir.to_string_lossy().to_string();
        }
        if !dir.pop() {
            break;
        }
    }
    env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}

fn extract_name(doc: &str) -> Option<String> {
    for line in doc.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("name:") {
            return Some(trimmed.trim_start_matches("name:").trim().to_string());
        }
    }
    None
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

async fn do_logs_summary(base_url: &str) {
    println!("Collecting logs from cluster...");
    let client = reqwest::Client::new();

    let logs = match client.get(&format!("{}/logs", base_url)).send().await {
        Ok(resp) => {
            if !resp.status().is_success() {
                eprintln!("Failed to get logs: {}", resp.text().await.unwrap_or_default());
                process::exit(1);
            }
            resp.text().await.unwrap_or_default()
        }
        Err(e) => {
            eprintln!("Failed to connect: {}", e);
            process::exit(1);
        }
    };

    if logs.trim().is_empty() || logs.trim() == "No logs found." {
        println!("No logs found.");
        return;
    }

    println!("Asking Claude to summarize...");

    let prompt = format!(
        "You are a Kubernetes SRE expert. Analyze the following pod logs from a cluster. \
         Summarize the findings: what is running, what is failing, why it is failing, \
         and what actions should be taken to fix it. Be concise and actionable.\n\n\
         LOGS:\n{}", logs
    );

    match run_claude(&prompt).await {
        Ok(summary) => println!("\n{}", summary),
        Err(e) => {
            eprintln!("Claude error: {}", e);
            process::exit(1);
        }
    }
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

async fn do_deploy() {
    let spec_path = find_project_dir() + "/specs/sre-agent-operator.yaml";
    if !std::path::Path::new(&spec_path).exists() {
        eprintln!("Spec not found: {}", spec_path);
        process::exit(1);
    }

    println!("Deploying sre-agent-operator to current cluster...");

    let output = Command::new("kubectl")
        .args(&["apply", "-f", &spec_path])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to run kubectl")
        .wait_with_output()
        .await
        .expect("Failed to wait for kubectl");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !stdout.is_empty() {
        println!("{}", stdout);
    }
    if !stderr.is_empty() {
        eprintln!("{}", stderr);
    }

    if !output.status.success() {
        eprintln!("Deploy failed.");
        process::exit(1);
    }

    println!("Waiting for sre-agent-operator pod to be ready...");
    loop {
        let check = Command::new("kubectl")
            .args(&["get", "pods", "-l", "app=sre-agent-operator", "-o", "jsonpath={.items[0].status.conditions[?(@.type==\"Ready\")].status}"])
            .output()
            .await;

        if let Ok(out) = check {
            let val = String::from_utf8_lossy(&out.stdout);
            if val.trim() == "True" {
                break;
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    println!("sre-agent-operator is ready.");
}

async fn do_k8s() {
    let cwd = env::current_dir().unwrap_or_default();

    let mut file_listing = String::new();
    let mut source_snippets = String::new();
    if let Ok(entries) = std::fs::read_dir(&cwd) {
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().to_string();
            if fname == "Containerfile" || fname == "Dockerfile" || fname == "specs" {
                continue;
            }
            file_listing.push_str(&format!("{}\n", fname));
            let ext = std::path::Path::new(&fname)
                .extension()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let is_source = matches!(ext.as_str(), "go" | "rs" | "py" | "js" | "ts" | "java" | "rb" | "mod" | "sum" | "toml" | "lock" | "json");
            if is_source {
                if let Ok(content) = std::fs::read_to_string(entry.path()) {
                    let preview: String = content.lines().take(50).collect::<Vec<&str>>().join("\n");
                    source_snippets.push_str(&format!("--- {} ---\n{}\n\n", fname, preview));
                }
            }
        }
    }

    let dir_name = cwd.file_name().unwrap_or_default().to_string_lossy().to_string();

    println!("Analyzing project source code...");
    let analyze_prompt = format!(
        "You are a devops expert. Analyze this project and determine:\n\
         1. A short app name suitable for a K8s deployment (lowercase, hyphens ok, no spaces)\n\
         2. The port the app listens on\n\n\
         Respond with EXACTLY two lines, nothing else:\n\
         name: <app-name>\n\
         port: <port-number>\n\n\
         Directory name: {}\n\nFILES:\n{}\n\nSOURCE:\n{}", dir_name, file_listing, source_snippets
    );

    let analysis = match run_claude(&analyze_prompt).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Claude error analyzing project: {}", e);
            process::exit(1);
        }
    };

    let mut name = String::new();
    let mut port: u16 = 8080;
    for line in analysis.lines() {
        let line = line.trim();
        if line.starts_with("name:") {
            name = line.trim_start_matches("name:").trim().to_string();
        } else if line.starts_with("port:") {
            port = line.trim_start_matches("port:").trim().parse().unwrap_or(8080);
        }
    }

    if name.is_empty() {
        name = dir_name.clone();
    }

    println!("Detected app: {} on port {}", name, port);

    let image_name = format!("{}:latest", name);

    println!("Asking Claude to generate Containerfile...");
    let containerfile_prompt = format!(
        "Generate a Containerfile for this project. The app listens on port {}. \
         Use multi-stage build. No comments. No blank lines between commands. \
         IMPORTANT: Output ONLY the raw Containerfile starting with FROM. \
         No markdown, no fences, no backticks, no explanation, no description. \
         Just the Containerfile instructions starting with FROM.\n\n\
         FILES:\n{}\n\nSOURCE:\n{}", port, file_listing, source_snippets
    );

    let containerfile_content = match run_claude(&containerfile_prompt).await {
        Ok(r) => extract_dockerfile(&r),
        Err(e) => {
            eprintln!("Claude error generating Containerfile: {}", e);
            process::exit(1);
        }
    };

    if containerfile_content.trim().is_empty() || !containerfile_content.trim().starts_with("FROM") {
        eprintln!("Claude returned invalid Containerfile (must start with FROM).");
        eprintln!("Raw response:\n{}", containerfile_content);
        process::exit(1);
    }

    let containerfile_path = cwd.join("Containerfile");
    std::fs::write(&containerfile_path, &containerfile_content).unwrap_or_else(|e| {
        eprintln!("Failed to write Containerfile: {}", e);
        process::exit(1);
    });
    println!("Generated: {}", containerfile_path.display());

    println!("Asking Claude to generate K8s manifests...");
    let k8s_prompt = format!(
        "Generate Kubernetes YAML manifests. \
         IMPORTANT: Output ONLY raw YAML starting with apiVersion. \
         No markdown, no fences, no backticks, no explanation, no description. \
         Just the raw YAML content.\n\n\
         Specs:\n\
         - Name: {}\n\
         - Image: {}\n\
         - imagePullPolicy: Never\n\
         - containerPort: {}\n\
         - replicas: 1\n\
         - namespace: default\n\n\
         Generate exactly:\n\
         1. A Deployment\n\
         2. A Service type LoadBalancer port {} targetPort {}\n\
         Separate with ---",
        name, image_name, port, port, port
    );

    let k8s_yaml = match run_claude(&k8s_prompt).await {
        Ok(r) => extract_k8s_yaml(&r),
        Err(e) => {
            eprintln!("Claude error generating K8s manifests: {}", e);
            process::exit(1);
        }
    };

    if k8s_yaml.trim().is_empty() || !k8s_yaml.contains("apiVersion") {
        eprintln!("Claude returned invalid K8s manifests (must contain apiVersion).");
        eprintln!("Raw response:\n{}", k8s_yaml);
        process::exit(1);
    }

    let specs_dir = cwd.join("specs");
    std::fs::create_dir_all(&specs_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create specs dir: {}", e);
        process::exit(1);
    });

    let spec_path = specs_dir.join(format!("{}.yaml", name));
    std::fs::write(&spec_path, &k8s_yaml).unwrap_or_else(|e| {
        eprintln!("Failed to write spec: {}", e);
        process::exit(1);
    });
    println!("Generated: {}", spec_path.display());

    let cluster_name = format!("{}-cluster", name);

    let start_sh = format!(
"#!/bin/bash\n\
set -e\n\
\n\
CLUSTER_NAME=\"{cluster}\"\n\
SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n\
\n\
kind create cluster --name \"$CLUSTER_NAME\"\n\
kubectl cluster-info --context \"kind-$CLUSTER_NAME\"\n\
\n\
cd \"$SCRIPT_DIR\"\n\
podman build -t {image} -f Containerfile .\n\
podman save {image} -o /tmp/{name}.tar\n\
kind load image-archive /tmp/{name}.tar --name \"$CLUSTER_NAME\"\n\
rm -f /tmp/{name}.tar\n\
\n\
for f in \"$SCRIPT_DIR/specs/\"*.yaml; do\n\
    echo \"Applying $f\"\n\
    kubectl apply -f \"$f\"\n\
done\n\
\n\
echo \"Waiting for {name} pod to be ready...\"\n\
while true; do\n\
    READY=$(kubectl get pods -l app={name} -o jsonpath='{{.items[0].status.conditions[?(@.type==\"Ready\")].status}}' 2>/dev/null)\n\
    if [ \"$READY\" = \"True\" ]; then\n\
        break\n\
    fi\n\
    sleep 1\n\
done\n\
\n\
echo \"{name} is running.\"\n\
kubectl get pods -A\n",
        cluster = cluster_name, image = image_name, name = name
    );

    let stop_sh = format!(
"#!/bin/bash\n\
set -e\n\
\n\
kind delete cluster --name \"{}\"\n",
        cluster_name
    );

    let start_path = cwd.join("start.sh");
    let stop_path = cwd.join("stop.sh");

    std::fs::write(&start_path, &start_sh).unwrap_or_else(|e| {
        eprintln!("Failed to write start.sh: {}", e);
        process::exit(1);
    });
    std::fs::write(&stop_path, &stop_sh).unwrap_or_else(|e| {
        eprintln!("Failed to write stop.sh: {}", e);
        process::exit(1);
    });

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&start_path, std::fs::Permissions::from_mode(0o755));
        let _ = std::fs::set_permissions(&stop_path, std::fs::Permissions::from_mode(0o755));
    }

    println!("Generated: {}", start_path.display());
    println!("Generated: {}", stop_path.display());
    println!("");
    println!("Run ./start.sh to create the cluster and deploy.");
    println!("Run ./stop.sh to tear down.");
}

fn extract_k8s_yaml(response: &str) -> String {
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
        let combined = blocks.join("\n---\n");
        if combined.contains("apiVersion") {
            return combined;
        }
    }

    let mut yaml_lines: Vec<&str> = Vec::new();
    let mut found = false;
    for line in response.lines() {
        if line.starts_with("apiVersion") {
            found = true;
        }
        if found {
            yaml_lines.push(line);
        }
    }

    if !yaml_lines.is_empty() {
        return yaml_lines.join("\n");
    }

    response.to_string()
}

fn extract_dockerfile(response: &str) -> String {
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
        for block in &blocks {
            if block.trim().starts_with("FROM") {
                return block.clone();
            }
        }
        return blocks[0].clone();
    }

    let mut from_lines: Vec<&str> = Vec::new();
    let mut found_from = false;
    for line in response.lines() {
        if line.starts_with("FROM") {
            found_from = true;
        }
        if found_from {
            from_lines.push(line);
        }
    }

    if !from_lines.is_empty() {
        return from_lines.join("\n");
    }

    response.to_string()
}


fn open_ui(base_url: &str) {
    println!("Opening UI at {}", base_url);
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(base_url).spawn();
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(base_url).spawn();
}

fn print_usage() {
    eprintln!("Usage: kovalski <command>");
    eprintln!("");
    eprintln!("Commands:");
    eprintln!("  logs         Read all pod logs from the cluster");
    eprintln!("  fix          Fix broken deployments using Claude AI");
    eprintln!("  status       Show all resources in the cluster (kubectl get all)");
    eprintln!("  logs-summary Summarize logs using Claude AI");
    eprintln!("  ui           Open the web UI in the browser");
    eprintln!("  deploy       Deploy sre-agent-operator to the current cluster");
    eprintln!("  k8s          Analyze project, generate Containerfile + K8s manifests, build, deploy");
    eprintln!("");
    eprintln!("Environment:");
    eprintln!("  KOVALSKI_URL  Base URL of the SRE agent (default: http://localhost:30080)");
}
