use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Planning,
    Working,
    Summarizing,
    Done,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepStatus {
    Pending,
    Running,
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub task: String,
    pub status: StepStatus,
    pub output: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunState {
    pub goal: String,
    pub provider: String,
    pub model: String,
    pub phase: Phase,
    pub steps: Vec<Step>,
    pub summary: String,
    pub error: String,
}

impl RunState {
    pub fn start(goal: &str, provider: &str, model: &str) -> Self {
        Self {
            goal: goal.to_string(),
            provider: provider.to_string(),
            model: model.to_string(),
            phase: Phase::Planning,
            steps: Vec::new(),
            summary: String::new(),
            error: String::new(),
        }
    }

    pub fn with_plan(self, tasks: &[String]) -> Self {
        let steps = tasks
            .iter()
            .map(|task| Step { task: task.clone(), status: StepStatus::Pending, output: String::new() })
            .collect();
        Self { phase: Phase::Working, steps, ..self }
    }

    fn with_step(self, index: usize, status: StepStatus, output: Option<String>) -> Self {
        let steps = self
            .steps
            .into_iter()
            .enumerate()
            .map(|(i, step)| match (i == index, &output) {
                (true, Some(text)) => Step { status, output: text.clone(), ..step },
                (true, None) => Step { status, ..step },
                _ => step,
            })
            .collect();
        Self { steps, ..self }
    }

    pub fn step_running(self, index: usize) -> Self {
        self.with_step(index, StepStatus::Running, None)
    }

    pub fn step_done(self, index: usize, output: &str) -> Self {
        self.with_step(index, StepStatus::Done, Some(output.to_string()))
    }

    pub fn summarizing(self) -> Self {
        Self { phase: Phase::Summarizing, ..self }
    }

    pub fn finished(self, summary: &str) -> Self {
        Self { phase: Phase::Done, summary: summary.to_string(), ..self }
    }

    pub fn failed(self, error: &str) -> Self {
        Self { phase: Phase::Failed, error: error.to_string(), ..self }
    }

    pub fn completed_work(&self) -> String {
        self.steps
            .iter()
            .filter(|step| step.status == StepStatus::Done)
            .enumerate()
            .map(|(i, step)| format!("Task {}: {}\nResult: {}", i + 1, step.task, step.output))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}
