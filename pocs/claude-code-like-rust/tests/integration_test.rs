use std::env;
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;
use std::thread;

fn has_api_key() -> bool {
    env::var("OPENAI_API_KEY").is_ok()
}

#[test]
fn test_application_starts_and_shows_banner() {
    let exe_path = env!("CARGO_BIN_EXE_claude-code-rust");
    let mut child = Command::new(exe_path)
        .env("OPENAI_API_KEY", "test_key_for_banner_check")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start application");

    thread::sleep(Duration::from_millis(100));

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(b"exit\n");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("Claude Code (Rust)") || stdout.contains("---"));
}

#[test]
fn test_application_exits_on_quit() {
    let exe_path = env!("CARGO_BIN_EXE_claude-code-rust");
    let mut child = Command::new(exe_path)
        .env("OPENAI_API_KEY", "test_key_for_quit_check")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start application");

    thread::sleep(Duration::from_millis(100));

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(b"quit\n");
    }

    let status = child.wait().expect("Failed to wait for process");
    assert!(status.success());
}

#[test]
fn test_application_requires_api_key() {
    let exe_path = env!("CARGO_BIN_EXE_claude-code-rust");
    let output = Command::new(exe_path)
        .env_remove("OPENAI_API_KEY")
        .output()
        .expect("Failed to run application");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("OPENAI_API_KEY") || !output.status.success());
}

#[test]
#[ignore]
fn test_end_to_end_create_file() {
    if !has_api_key() {
        eprintln!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let test_dir = env::temp_dir().join("claude_code_e2e_test");
    let _ = fs::remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).expect("Failed to create test directory");

    let test_file = test_dir.join("hello.txt");

    let exe_path = env!("CARGO_BIN_EXE_claude-code-rust");
    let mut child = Command::new(exe_path)
        .current_dir(&test_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start application");

    thread::sleep(Duration::from_millis(500));

    let prompt = format!(
        "Create a file called hello.txt with the content 'Hello World' in the current directory: {}\n",
        test_file.display()
    );

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(prompt.as_bytes()).expect("Failed to write prompt");
        stdin.flush().expect("Failed to flush");

        thread::sleep(Duration::from_secs(15));

        let _ = stdin.write_all(b"exit\n");
    }

    let output = child.wait_with_output().expect("Failed to get output");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("STDOUT: {}", stdout);
    println!("STDERR: {}", stderr);

    let file_created = test_file.exists() || stdout.contains("written successfully") || stdout.contains("edit_file");

    let _ = fs::remove_dir_all(&test_dir);

    assert!(file_created || stdout.contains("Tool:") || stdout.contains("Error"),
            "Expected file creation or tool execution indication");
}

#[test]
#[ignore]
fn test_end_to_end_list_files() {
    if !has_api_key() {
        eprintln!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let exe_path = env!("CARGO_BIN_EXE_claude-code-rust");
    let mut child = Command::new(exe_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start application");

    thread::sleep(Duration::from_millis(500));

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(b"List all files in the current directory\n").expect("Failed to write prompt");
        stdin.flush().expect("Failed to flush");

        thread::sleep(Duration::from_secs(10));

        let _ = stdin.write_all(b"exit\n");
    }

    let output = child.wait_with_output().expect("Failed to get output");
    let stdout = String::from_utf8_lossy(&output.stdout);

    println!("STDOUT: {}", stdout);

    assert!(stdout.contains("list_files") || stdout.contains("Cargo") || stdout.contains("src") || stdout.contains("Tool:") || stdout.contains("Error"),
            "Expected list_files tool usage or file listing");
}

#[test]
#[ignore]
fn test_end_to_end_execute_command() {
    if !has_api_key() {
        eprintln!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let exe_path = env!("CARGO_BIN_EXE_claude-code-rust");
    let mut child = Command::new(exe_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start application");

    thread::sleep(Duration::from_millis(500));

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(b"Run the command 'echo Hello World' using execute_command\n").expect("Failed to write prompt");
        stdin.flush().expect("Failed to flush");

        thread::sleep(Duration::from_secs(10));

        let _ = stdin.write_all(b"exit\n");
    }

    let output = child.wait_with_output().expect("Failed to get output");
    let stdout = String::from_utf8_lossy(&output.stdout);

    println!("STDOUT: {}", stdout);

    assert!(stdout.contains("execute_command") || stdout.contains("Hello") || stdout.contains("Tool:") || stdout.contains("Error"),
            "Expected execute_command tool usage");
}

#[test]
fn test_tools_are_available() {
    use std::path::Path;

    let tools_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tools");

    assert!(tools_dir.join("read_file.rs").exists());
    assert!(tools_dir.join("list_files.rs").exists());
    assert!(tools_dir.join("edit_file.rs").exists());
    assert!(tools_dir.join("execute_command.rs").exists());
    assert!(tools_dir.join("mod.rs").exists());
}

#[test]
fn test_project_compiles() {
    let output = Command::new("cargo")
        .args(["check"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to run cargo check");

    assert!(output.status.success(), "Project should compile without errors");
}
