use crate::agents::{ProcessRunner, call};
use crate::config::{AgentConfig, process_env, read_config};
use crate::error::AppError;
use crate::plan::parse_plan;
use crate::prompts::{PromptParams, load_prompt};
use crate::run_state::RunState;
use restate_sdk::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const STATE: &str = "state";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRequest {
    pub goal: String,
}

fn retry() -> RunRetryPolicy {
    RunRetryPolicy::new()
        .initial_delay(Duration::from_secs(2))
        .exponentiation_factor(2.0)
        .max_attempts(3)
}

async fn ask(config: AgentConfig, prompt: Result<String, AppError>) -> Result<String, AppError> {
    let output = call(config.provider.agent().as_ref(), &ProcessRunner, &config.model, &prompt?).await?;
    Ok(output.trim().to_string())
}

fn params<'a>(pairs: &[(&'a str, &str)]) -> PromptParams<'a> {
    pairs.iter().map(|(name, value)| (*name, value.to_string())).collect()
}

fn save(ctx: &WorkflowContext<'_>, state: &RunState) {
    ctx.set(STATE, Json(state.clone()));
}

async fn run_steps(ctx: &WorkflowContext<'_>, config: &AgentConfig, goal: &str, state: &mut RunState) -> Result<String, TerminalError> {
    let plan_config = config.clone();
    let plan_prompt = load_prompt("plan-goal", &params(&[("goal", goal)]));
    let tasks = ctx
        .run(|| async move { Ok(Json(parse_plan(&ask(plan_config, plan_prompt).await?)?)) })
        .name("plan")
        .retry_policy(retry())
        .await?
        .into_inner();
    *state = state.clone().with_plan(&tasks);
    save(ctx, state);
    for (index, task) in tasks.iter().enumerate() {
        *state = state.clone().step_running(index);
        save(ctx, state);
        let done = state.completed_work();
        let previous = if done.is_empty() { "none yet".to_string() } else { done };
        let prompt = load_prompt("execute-task", &params(&[("goal", goal), ("task", task), ("previous_work", &previous)]));
        let task_config = config.clone();
        let output = ctx
            .run(|| async move { Ok(ask(task_config, prompt).await?) })
            .name(format!("task {}", index + 1))
            .retry_policy(retry())
            .await?;
        *state = state.clone().step_done(index, &output);
        save(ctx, state);
    }
    *state = state.clone().summarizing();
    save(ctx, state);
    let prompt = load_prompt("summarize-run", &params(&[("goal", goal), ("work", &state.completed_work())]));
    let summary_config = config.clone();
    let summary = ctx
        .run(|| async move { Ok(ask(summary_config, prompt).await?) })
        .name("summarize")
        .retry_policy(retry())
        .await?;
    *state = state.clone().finished(&summary);
    save(ctx, state);
    Ok(summary)
}

pub struct AgentWorkflow;

#[restate_sdk::workflow]
impl AgentWorkflow {
    #[handler]
    async fn run(&self, ctx: WorkflowContext<'_>, request: Json<RunRequest>) -> Result<String, HandlerError> {
        let goal = request.into_inner().goal.trim().to_string();
        if goal.is_empty() {
            return Err(TerminalError::new_with_code(400, "goal is required").into());
        }
        let config = ctx
            .run(|| async { Ok(Json(read_config(process_env)?)) })
            .name("read config")
            .await?
            .into_inner();
        let mut state = RunState::start(&goal, config.provider.name(), &config.model);
        save(&ctx, &state);
        match run_steps(&ctx, &config, &goal, &mut state).await {
            Ok(summary) => Ok(summary),
            Err(error) => {
                save(&ctx, &state.clone().failed(error.message()));
                Err(error.into())
            }
        }
    }

    #[handler]
    async fn status(&self, ctx: SharedWorkflowContext<'_>) -> Result<Json<Option<RunState>>, HandlerError> {
        let state = ctx.get::<Json<RunState>>(STATE).await?;
        Ok(Json(state.map(Json::into_inner)))
    }
}
