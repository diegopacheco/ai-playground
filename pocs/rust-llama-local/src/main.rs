use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::io::{self, Write};
use std::num::NonZeroU32;

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/llama-3.gguf".to_string());

    let backend = LlamaBackend::init().expect("Failed to initialize backend");
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .expect("Failed to load model");

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048));

    println!("Llama 3 Chat - Type 'quit' to exit");
    println!("-----------------------------------");

    let mut seed: u32 = 42;

    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input == "quit" {
            break;
        }

        if input.is_empty() {
            continue;
        }

        let mut ctx = model
            .new_context(&backend, ctx_params.clone())
            .expect("Failed to create context");

        let prompt = format!("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", input);
        let tokens = model
            .str_to_token(&prompt, AddBos::Never)
            .expect("Failed to tokenize");

        let mut batch = LlamaBatch::new(2048, 1);
        for (i, token) in tokens.iter().enumerate() {
            batch.add(*token, i as i32, &[0], i == tokens.len() - 1).unwrap();
        }

        ctx.decode(&mut batch).expect("Failed to decode");

        print!("Assistant: ");
        io::stdout().flush().unwrap();

        let mut n_cur = tokens.len();
        let n_len = 512;

        while n_cur < n_len {
            let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
            let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
            seed = seed.wrapping_add(1);
            let new_token = candidates_p.sample_token(seed);

            if model.is_eog_token(new_token) {
                break;
            }

            if let Ok(token_str) = model.token_to_str(new_token, Special::Tokenize) {
                print!("{}", token_str);
                io::stdout().flush().unwrap();
            }

            batch.clear();
            batch.add(new_token, n_cur as i32, &[0], true).unwrap();
            n_cur += 1;

            ctx.decode(&mut batch).expect("Failed to decode");
        }

        println!("\n");
    }

    println!("Goodbye!");
}
