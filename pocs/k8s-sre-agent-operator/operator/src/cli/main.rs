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
        "k8s" => rt.block_on(do_k8s(&args[2..])),
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

async fn do_k8s(args: &[String]) {
    let mut name = String::new();
    let mut image = String::new();
    let mut port: u16 = 8080;
    let mut replicas: u32 = 1;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => { i += 1; name = args[i].clone(); }
            "--image" => { i += 1; image = args[i].clone(); }
            "--port" => { i += 1; port = args[i].parse().unwrap_or(8080); }
            "--replicas" => { i += 1; replicas = args[i].parse().unwrap_or(1); }
            _ => {
                eprintln!("Unknown k8s flag: {}", args[i]);
                eprintln!("Usage: kovalski k8s --name <name> --image <image> [--port <port>] [--replicas <n>]");
                process::exit(1);
            }
        }
        i += 1;
    }

    if name.is_empty() || image.is_empty() {
        eprintln!("Usage: kovalski k8s --name <name> --image <image> [--port <port>] [--replicas <n>]");
        process::exit(1);
    }

    let yaml = format!(
"apiVersion: apps/v1
kind: Deployment
metadata:
  name: {}
  namespace: default
spec:
  replicas: {}
  selector:
    matchLabels:
      app: {}
  template:
    metadata:
      labels:
        app: {}
    spec:
      containers:
        - name: {}
          image: {}
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: {}
---
apiVersion: v1
kind: Service
metadata:
  name: {}
  namespace: default
spec:
  type: LoadBalancer
  selector:
    app: {}
  ports:
    - port: {}
      targetPort: {}
      protocol: TCP",
        name, replicas, name, name, name, image, port, name, name, port, port
    );

    let specs_dir = env::current_dir()
        .unwrap_or_default()
        .join("specs");
    std::fs::create_dir_all(&specs_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create specs dir: {}", e);
        process::exit(1);
    });

    let spec_path = specs_dir.join(format!("{}.yaml", name));
    std::fs::write(&spec_path, &yaml).unwrap_or_else(|e| {
        eprintln!("Failed to write spec: {}", e);
        process::exit(1);
    });
    println!("Generated: {}", spec_path.display());

    println!("Applying to cluster...");
    let output = Command::new("kubectl")
        .args(&["apply", "-f", &spec_path.to_string_lossy()])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to run kubectl")
        .wait_with_output()
        .await
        .expect("Failed to wait for kubectl");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stdout.is_empty() { println!("{}", stdout); }
    if !stderr.is_empty() { eprintln!("{}", stderr); }

    if !output.status.success() {
        eprintln!("Apply failed.");
        process::exit(1);
    }

    println!("Waiting for {} to be ready...", name);
    loop {
        let check = Command::new("kubectl")
            .args(&["get", "pods", "-l", &format!("app={}", name), "-o", "jsonpath={.items[0].status.conditions[?(@.type==\"Ready\")].status}"])
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

    println!("{} is ready.", name);

    let svc_output = Command::new("kubectl")
        .args(&["get", "svc", &name, "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"])
        .output()
        .await;

    if let Ok(out) = svc_output {
        let ip = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !ip.is_empty() {
            println!("Service available at: http://{}:{}", ip, port);
        } else {
            println!("Service created (LoadBalancer IP pending).");
        }
    }
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
    eprintln!("  k8s          Generate K8s manifests, save to specs/, and apply");
    eprintln!("               --name <name> --image <image> [--port <port>] [--replicas <n>]");
    eprintln!("");
    eprintln!("Environment:");
    eprintln!("  KOVALSKI_URL  Base URL of the SRE agent (default: http://localhost:30080)");
}
