use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRequest {
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    pub id: String,
    pub title: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingContent {
    pub title: String,
    pub topics: Vec<Topic>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionRequest {
    pub question: String,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResponse {
    pub answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuizQuestion {
    pub id: usize,
    pub question: String,
    pub options: Vec<String>,
    pub correct_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quiz {
    pub questions: Vec<QuizQuestion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuizSubmission {
    pub answers: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuizResult {
    pub score: usize,
    pub total: usize,
    pub percentage: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub id: String,
    pub user_name: String,
    pub training_title: String,
    pub score: usize,
    pub total: usize,
    pub percentage: f64,
    pub date: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateRequest {
    pub user_name: String,
    pub training_title: String,
    pub score: usize,
    pub total: usize,
    pub percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SseEvent {
    #[serde(rename = "start")]
    Start { message: String, total_steps: usize },
    #[serde(rename = "progress")]
    Progress { step: usize, message: String },
    #[serde(rename = "training_ready")]
    TrainingReady { training: TrainingContent },
    #[serde(rename = "quiz_ready")]
    QuizReady { quiz: Quiz },
    #[serde(rename = "error")]
    Error { message: String },
}
