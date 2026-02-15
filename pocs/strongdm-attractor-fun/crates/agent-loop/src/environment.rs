use std::path::PathBuf;

pub struct ExecutionEnvironment {
    pub working_dir: PathBuf,
    pub allowed_paths: Vec<PathBuf>,
    pub shell: String,
}

impl Default for ExecutionEnvironment {
    fn default() -> Self {
        Self {
            working_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            allowed_paths: vec![],
            shell: "sh".to_string(),
        }
    }
}

impl ExecutionEnvironment {
    pub fn new(working_dir: PathBuf) -> Self {
        Self {
            working_dir,
            ..Default::default()
        }
    }

    pub fn is_path_allowed(&self, path: &std::path::Path) -> bool {
        if self.allowed_paths.is_empty() {
            return true;
        }
        self.allowed_paths.iter().any(|p| path.starts_with(p))
    }
}
