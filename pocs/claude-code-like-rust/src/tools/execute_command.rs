use std::process::Command;

pub fn execute_command(program: &str, args: &[String]) -> String {
    if program.is_empty() {
        return "Error: program name cannot be empty".to_string();
    }
    match Command::new(program).args(args).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            if output.status.success() {
                if stdout.is_empty() && stderr.is_empty() {
                    format!("Command executed successfully with exit code 0")
                } else if stderr.is_empty() {
                    stdout.to_string()
                } else {
                    format!("{}\n{}", stdout, stderr)
                }
            } else {
                let code = output.status.code().unwrap_or(-1);
                if stderr.is_empty() {
                    format!("Command failed with exit code {}\n{}", code, stdout)
                } else {
                    format!("Command failed with exit code {}\n{}\n{}", code, stdout, stderr)
                }
            }
        }
        Err(e) => format!("Error executing command: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_command_echo() {
        let result = execute_command("echo", &["Hello".to_string()]);
        assert!(result.contains("Hello"));
    }

    #[test]
    fn test_execute_command_with_multiple_args() {
        let result = execute_command("echo", &["arg1".to_string(), "arg2".to_string(), "arg3".to_string()]);
        assert!(result.contains("arg1"));
        assert!(result.contains("arg2"));
        assert!(result.contains("arg3"));
    }

    #[test]
    fn test_execute_command_no_args() {
        let result = execute_command("echo", &[]);
        assert!(!result.starts_with("Error"));
    }

    #[test]
    fn test_execute_command_empty_program() {
        let result = execute_command("", &[]);
        assert_eq!(result, "Error: program name cannot be empty");
    }

    #[test]
    fn test_execute_command_nonexistent_program() {
        let result = execute_command("nonexistent_program_12345", &[]);
        assert!(result.starts_with("Error executing command:"));
    }

    #[test]
    fn test_execute_command_pwd() {
        let result = execute_command("pwd", &[]);
        assert!(!result.is_empty());
        assert!(!result.starts_with("Error"));
    }

    #[test]
    fn test_execute_command_ls() {
        let result = execute_command("ls", &[".".to_string()]);
        assert!(!result.starts_with("Error"));
    }

    #[test]
    fn test_execute_command_failing_command() {
        let result = execute_command("ls", &["/nonexistent/path/12345".to_string()]);
        assert!(result.contains("failed") || result.contains("Error") || result.contains("No such file"));
    }

    #[test]
    fn test_execute_command_true() {
        let result = execute_command("true", &[]);
        assert!(result.contains("executed successfully") || result.is_empty() || !result.contains("Error"));
    }

    #[test]
    fn test_execute_command_false() {
        let result = execute_command("false", &[]);
        assert!(result.contains("failed") || result.contains("exit code"));
    }
}
