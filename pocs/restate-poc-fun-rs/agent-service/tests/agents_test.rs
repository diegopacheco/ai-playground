use agent_service::agents::{CommandRunner, Provider, call, command_path};
use agent_service::config::{AgentConfig, read_config, service_port};
use agent_service::error::AppError;
use std::path::Path;
use std::sync::Mutex;

#[derive(Default)]
struct Recorder {
    calls: Mutex<Vec<Vec<String>>>,
}

impl CommandRunner for Recorder {
    async fn run(&self, command: Vec<String>) -> Result<String, AppError> {
        self.calls.lock().unwrap().push(command);
        Ok("ok".to_string())
    }
}

fn env_of(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
    move |name| pairs.iter().find(|(key, _)| *key == name).map(|(_, value)| value.to_string())
}

#[tokio::test]
async fn each_provider_shells_out_to_its_own_cli_with_the_model_and_prompt() {
    let expected = [
        (Provider::Claude, vec!["claude", "-p", "--model", "m", "hi"]),
        (Provider::Codex, vec!["codex", "exec", "--model", "m", "hi"]),
        (Provider::Agy, vec!["agy", "-p", "--model", "m", "hi"]),
        (Provider::Ollama, vec!["ollama", "run", "m", "hi"]),
    ];
    for (provider, command) in expected {
        let recorder = Recorder::default();
        call(provider.agent().as_ref(), &recorder, "m", "hi").await.unwrap();
        let calls = recorder.calls.lock().unwrap().clone();
        assert_eq!(calls, vec![command.iter().map(|part| part.to_string()).collect::<Vec<_>>()]);
    }
}

#[tokio::test]
async fn an_empty_model_or_prompt_is_rejected_before_any_cli_runs() {
    let recorder = Recorder::default();
    let agent = Provider::Claude.agent();
    assert!(call(agent.as_ref(), &recorder, " ", "hi").await.unwrap_err().0.contains("model"));
    assert!(call(agent.as_ref(), &recorder, "m", "").await.unwrap_err().0.contains("prompt"));
    assert!(recorder.calls.lock().unwrap().is_empty());
}

#[test]
fn config_refuses_an_unknown_provider_instead_of_failing_later_inside_the_workflow() {
    assert!(read_config(env_of(&[("AGENT_PROVIDER", "gpt"), ("AGENT_MODEL", "m")])).unwrap_err().0.contains("AGENT_PROVIDER"));
    assert!(read_config(env_of(&[("AGENT_PROVIDER", "ollama"), ("AGENT_MODEL", "")])).unwrap_err().0.contains("AGENT_MODEL"));
    assert_eq!(
        read_config(env_of(&[("AGENT_PROVIDER", "ollama"), ("AGENT_MODEL", "llama3.2")])).unwrap(),
        AgentConfig { provider: Provider::Ollama, model: "llama3.2".to_string() }
    );
}

#[test]
fn the_journaled_config_round_trips_with_the_provider_name_restate_stores() {
    let config = AgentConfig { provider: Provider::Codex, model: "m".to_string() };
    let json = serde_json::to_string(&config).unwrap();
    assert_eq!(json, r#"{"provider":"codex","model":"m"}"#);
    assert_eq!(serde_json::from_str::<AgentConfig>(&json).unwrap(), config);
}

#[test]
fn service_port_must_be_a_real_port_so_restate_can_register_the_deployment() {
    assert!(service_port(env_of(&[])).is_err());
    assert!(service_port(env_of(&[("AGENT_SERVICE_PORT", "0")])).is_err());
    assert!(service_port(env_of(&[("AGENT_SERVICE_PORT", "abc")])).is_err());
    assert_eq!(service_port(env_of(&[("AGENT_SERVICE_PORT", "9180")])).unwrap(), 9180);
}

#[test]
fn cli_lookup_path_puts_user_bin_dirs_first_and_keeps_the_inherited_path_without_duplicates() {
    let path = command_path(Path::new("/h"), "/usr/bin:/h/.local/bin");
    let parts: Vec<&str> = path.split(':').collect();
    assert_eq!(parts[0], "/h/.local/bin");
    assert!(parts.contains(&"/usr/bin"));
    assert_eq!(parts.iter().filter(|part| **part == "/h/.local/bin").count(), 1);
}
