use std::env;
use std::process;

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
        "fix" => rt.block_on(do_post(&format!("{}/fix", base_url))),
        "status" => rt.block_on(do_get(&format!("{}/status", base_url))),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
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

async fn do_post(url: &str) {
    let client = reqwest::Client::new();
    match client.post(url).send().await {
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
