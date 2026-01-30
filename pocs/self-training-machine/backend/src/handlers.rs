use axum::{
    extract::Json,
    response::sse::{Event, Sse},
};
use futures::stream::{self, Stream};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::claude::{answer_question, generate_quiz, generate_training};
use crate::models::{
    Certificate, CertificateRequest, QuestionRequest, QuestionResponse, Quiz, QuizResult,
    QuizSubmission, SseEvent, TrainingContent, TrainingRequest,
};

pub struct AppState {
    pub current_training: Mutex<Option<TrainingContent>>,
    pub current_quiz: Mutex<Option<Quiz>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            current_training: Mutex::new(None),
            current_quiz: Mutex::new(None),
        }
    }
}

pub async fn health() -> &'static str {
    "OK"
}

pub async fn generate_training_handler(
    state: Arc<AppState>,
    Json(request): Json<TrainingRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let prompt = request.prompt;

    let stream = stream::unfold(
        (0, prompt, state, None::<TrainingContent>, None::<Quiz>),
        |(step, prompt, state, training, quiz)| async move {
            match step {
                0 => {
                    let event = SseEvent::Start {
                        message: "Starting training generation...".to_string(),
                        total_steps: 3,
                    };
                    let data = serde_json::to_string(&event).unwrap();
                    Some((
                        Ok(Event::default().data(data)),
                        (1, prompt, state, training, quiz),
                    ))
                }
                1 => {
                    let event = SseEvent::Progress {
                        step: 1,
                        message: "Generating training content with Claude Opus 4.5...".to_string(),
                    };
                    let data = serde_json::to_string(&event).unwrap();
                    Some((
                        Ok(Event::default().data(data)),
                        (2, prompt, state, training, quiz),
                    ))
                }
                2 => match generate_training(&prompt).await {
                    Ok(training_content) => {
                        let mut guard = state.current_training.lock().await;
                        *guard = Some(training_content.clone());
                        drop(guard);

                        let event = SseEvent::TrainingReady {
                            training: training_content.clone(),
                        };
                        let data = serde_json::to_string(&event).unwrap();
                        Some((
                            Ok(Event::default().data(data)),
                            (3, prompt, state, Some(training_content), quiz),
                        ))
                    }
                    Err(e) => {
                        let event = SseEvent::Error {
                            message: format!("Failed to generate training: {}", e),
                        };
                        let data = serde_json::to_string(&event).unwrap();
                        Some((Ok(Event::default().data(data)), (99, prompt, state, training, quiz)))
                    }
                },
                3 => {
                    let event = SseEvent::Progress {
                        step: 2,
                        message: "Generating quiz questions...".to_string(),
                    };
                    let data = serde_json::to_string(&event).unwrap();
                    Some((
                        Ok(Event::default().data(data)),
                        (4, prompt, state, training, quiz),
                    ))
                }
                4 => {
                    if let Some(ref t) = training {
                        match generate_quiz(t).await {
                            Ok(quiz_content) => {
                                let mut guard = state.current_quiz.lock().await;
                                *guard = Some(quiz_content.clone());
                                drop(guard);

                                let event = SseEvent::QuizReady {
                                    quiz: quiz_content.clone(),
                                };
                                let data = serde_json::to_string(&event).unwrap();
                                Some((
                                    Ok(Event::default().data(data)),
                                    (5, prompt, state, training, Some(quiz_content)),
                                ))
                            }
                            Err(e) => {
                                let event = SseEvent::Error {
                                    message: format!("Failed to generate quiz: {}", e),
                                };
                                let data = serde_json::to_string(&event).unwrap();
                                Some((Ok(Event::default().data(data)), (99, prompt, state, training, quiz)))
                            }
                        }
                    } else {
                        None
                    }
                }
                5 => {
                    let event = SseEvent::Progress {
                        step: 3,
                        message: "Training and quiz ready!".to_string(),
                    };
                    let data = serde_json::to_string(&event).unwrap();
                    Some((
                        Ok(Event::default().data(data)),
                        (99, prompt, state, training, quiz),
                    ))
                }
                _ => None,
            }
        },
    );

    Sse::new(stream)
}

pub async fn ask_question_handler(
    state: Arc<AppState>,
    Json(request): Json<QuestionRequest>,
) -> Json<QuestionResponse> {
    let training_guard = state.current_training.lock().await;
    let context = if let Some(ref training) = *training_guard {
        training
            .topics
            .iter()
            .map(|t| format!("{}\n{}", t.title, t.content))
            .collect::<Vec<_>>()
            .join("\n\n")
    } else {
        request.context
    };
    drop(training_guard);

    match answer_question(&request.question, &context).await {
        Ok(answer) => Json(QuestionResponse { answer }),
        Err(e) => Json(QuestionResponse {
            answer: format!("Sorry, I couldn't answer that question: {}", e),
        }),
    }
}

pub async fn submit_quiz_handler(
    state: Arc<AppState>,
    Json(submission): Json<QuizSubmission>,
) -> Json<QuizResult> {
    let quiz_guard = state.current_quiz.lock().await;

    if let Some(ref quiz) = *quiz_guard {
        let mut correct = 0;
        let total = quiz.questions.len();

        for (i, answer) in submission.answers.iter().enumerate() {
            if i < quiz.questions.len() && *answer == quiz.questions[i].correct_index {
                correct += 1;
            }
        }

        let percentage = (correct as f64 / total as f64) * 100.0;
        let passed = percentage >= 70.0;

        Json(QuizResult {
            score: correct,
            total,
            percentage,
            passed,
        })
    } else {
        Json(QuizResult {
            score: 0,
            total: 0,
            percentage: 0.0,
            passed: false,
        })
    }
}

pub async fn generate_certificate_handler(
    Json(request): Json<CertificateRequest>,
) -> Json<Certificate> {
    let id = uuid::Uuid::new_v4().to_string();
    let date = chrono_lite_date();

    Json(Certificate {
        id,
        user_name: request.user_name,
        training_title: request.training_title,
        score: request.score,
        total: request.total,
        percentage: request.percentage,
        date,
    })
}

fn chrono_lite_date() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = duration.as_secs();
    let days = secs / 86400;
    let years = 1970 + days / 365;
    let remaining_days = days % 365;
    let month = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    format!("{:04}-{:02}-{:02}", years, month.min(12), day.min(31))
}
