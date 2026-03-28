use std::path::Path;
use std::fs;

#[derive(Debug, Clone)]
pub enum ProjectType {
    Rust,
    Go,
    NodeTs,
    JavaMaven,
    JavaGradle,
    Unknown,
}

impl std::fmt::Display for ProjectType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProjectType::Rust => write!(f, "Rust"),
            ProjectType::Go => write!(f, "Go"),
            ProjectType::NodeTs => write!(f, "Node/TypeScript"),
            ProjectType::JavaMaven => write!(f, "Java (Maven)"),
            ProjectType::JavaGradle => write!(f, "Java (Gradle)"),
            ProjectType::Unknown => write!(f, "Unknown"),
        }
    }
}

pub struct DetectedProject {
    pub project_type: ProjectType,
    pub project_root: String,
}

fn detect_at(base: &Path) -> Option<ProjectType> {
    if base.join("Cargo.toml").exists() {
        Some(ProjectType::Rust)
    } else if base.join("go.mod").exists() {
        Some(ProjectType::Go)
    } else if base.join("package.json").exists() {
        Some(ProjectType::NodeTs)
    } else if base.join("pom.xml").exists() {
        Some(ProjectType::JavaMaven)
    } else if base.join("build.gradle").exists() {
        Some(ProjectType::JavaGradle)
    } else {
        None
    }
}

pub fn detect_project(path: &str) -> DetectedProject {
    let base = Path::new(path);
    if let Some(pt) = detect_at(base) {
        return DetectedProject {
            project_type: pt,
            project_root: path.to_string(),
        };
    }
    if let Ok(entries) = fs::read_dir(base) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if !entry_path.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if name == ".git" || name == "node_modules" || name == "target" {
                continue;
            }
            if let Some(pt) = detect_at(&entry_path) {
                return DetectedProject {
                    project_type: pt,
                    project_root: entry_path.to_string_lossy().to_string(),
                };
            }
        }
    }
    DetectedProject {
        project_type: ProjectType::Unknown,
        project_root: path.to_string(),
    }
}

pub fn build_command(project: &ProjectType) -> (&str, Vec<&str>) {
    match project {
        ProjectType::Rust => ("cargo", vec!["build"]),
        ProjectType::Go => ("go", vec!["build", "./..."]),
        ProjectType::NodeTs => ("npm", vec!["run", "build"]),
        ProjectType::JavaMaven => ("mvn", vec!["compile"]),
        ProjectType::JavaGradle => ("gradle", vec!["build"]),
        ProjectType::Unknown => ("echo", vec!["unknown project type"]),
    }
}

pub fn test_command(project: &ProjectType) -> (&str, Vec<&str>) {
    match project {
        ProjectType::Rust => ("cargo", vec!["test"]),
        ProjectType::Go => ("go", vec!["test", "./..."]),
        ProjectType::NodeTs => ("npm", vec!["test"]),
        ProjectType::JavaMaven => ("mvn", vec!["test"]),
        ProjectType::JavaGradle => ("gradle", vec!["test"]),
        ProjectType::Unknown => ("echo", vec!["unknown project type"]),
    }
}
