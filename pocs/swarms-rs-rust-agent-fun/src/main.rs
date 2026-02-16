use std::env;
use swarms_rs::{
    agent::SwarmsAgentBuilder,
    llm::provider::openai::OpenAI,
    structs::{agent::Agent, tool::Tool},
};

#[derive(Debug)]
struct CalculatorTool;

#[derive(Debug, serde::Deserialize)]
struct CalculatorArgs {
    expression: String,
}

impl Tool for CalculatorTool {
    type Error = std::io::Error;
    type Args = CalculatorArgs;
    type Output = String;
    const NAME: &'static str = "calculator";

    fn definition(&self) -> swarms_rs::llm::request::ToolDefinition {
        swarms_rs::llm::request::ToolDefinition {
            name: "calculator".to_string(),
            description: "Evaluate mathematical expressions".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        match args.expression.as_str() {
            "2+2" => Ok("4".to_string()),
            "10*5" => Ok("50".to_string()),
            "15/3" => Ok("5".to_string()),
            "7-3" => Ok("4".to_string()),
            _ => Ok(format!("Result of {}: computed", args.expression)),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    env_logger::init();

    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = OpenAI::from_url("https://api.openai.com/v1", &api_key).set_model("gpt-4o-mini");

    let agent = SwarmsAgentBuilder::new_with_model(client)
        .agent_name("RustAgent")
        .system_prompt(
            "You are a helpful assistant. When asked to calculate something, use the calculator tool. \
             Keep your responses clear and concise."
        )
        .user_name("User")
        .max_loops(2)
        .temperature(0.3)
        .add_tool(CalculatorTool)
        .build();

    let result = agent
        .run("Calculate 2+2, then 10*5, and explain your reasoning".to_string())
        .await?;

    println!("Result: {}", result);

    Ok(())
}
