use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::num::NonZeroU32;
use std::path::Path;
use std::process::exit;

const N_CTX: u32 = 8192;
const MAX_ANSWER_TOKENS: usize = 512;
const MAX_DOC_CHARS: usize = 12000;

fn escape_json(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 16);
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn fail(message: &str) -> ! {
    println!("{{\"ok\":false,\"error\":\"{}\"}}", escape_json(message));
    exit(1);
}

fn extract_text(pdf_path: &str) -> String {
    match pdf_extract::extract_text(Path::new(pdf_path)) {
        Ok(text) => text,
        Err(err) => fail(&format!("pdf extraction failed: {}", err)),
    }
}

fn clean(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<&str>>()
        .join("\n")
}

fn truncate(text: &str) -> String {
    match text.char_indices().nth(MAX_DOC_CHARS) {
        Some((cut, _)) => text[..cut].to_string(),
        None => text.to_string(),
    }
}

fn build_prompt(document: &str, question: &str) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
         You answer strictly from the document supplied by the user. \
         If the document does not contain the answer, say so.\
         <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
         DOCUMENT:\n{}\n\nQUESTION: {}\
         <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        document, question
    )
}

fn generate(model_path: &str, prompt: &str) -> String {
    let backend = match LlamaBackend::init() {
        Ok(backend) => backend,
        Err(err) => fail(&format!("llama backend init failed: {}", err)),
    };

    let model = match LlamaModel::load_from_file(&backend, model_path, &LlamaModelParams::default())
    {
        Ok(model) => model,
        Err(err) => fail(&format!("model load failed ({}): {}", model_path, err)),
    };

    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(N_CTX));
    let mut ctx = match model.new_context(&backend, ctx_params) {
        Ok(ctx) => ctx,
        Err(err) => fail(&format!("context creation failed: {}", err)),
    };

    let tokens = match model.str_to_token(prompt, AddBos::Never) {
        Ok(tokens) => tokens,
        Err(err) => fail(&format!("tokenization failed: {}", err)),
    };

    if tokens.len() + MAX_ANSWER_TOKENS >= N_CTX as usize {
        fail("document plus question exceeds the model context window");
    }

    let mut batch = LlamaBatch::new(N_CTX as usize, 1);
    for (i, token) in tokens.iter().enumerate() {
        if batch.add(*token, i as i32, &[0], i == tokens.len() - 1).is_err() {
            fail("failed to fill the prompt batch");
        }
    }

    if let Err(err) = ctx.decode(&mut batch) {
        fail(&format!("prompt decode failed: {}", err));
    }

    let mut answer = String::new();
    let mut position = tokens.len();
    let mut seed: u32 = 42;
    let limit = position + MAX_ANSWER_TOKENS;

    while position < limit {
        let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
        let mut data = LlamaTokenDataArray::from_iter(candidates, false);
        seed = seed.wrapping_add(1);
        let next = data.sample_token(seed);

        if model.is_eog_token(next) {
            break;
        }

        if let Ok(piece) = model.token_to_str(next, Special::Tokenize) {
            answer.push_str(&piece);
        }

        batch.clear();
        if batch.add(next, position as i32, &[0], true).is_err() {
            break;
        }
        position += 1;

        if let Err(err) = ctx.decode(&mut batch) {
            fail(&format!("decode failed: {}", err));
        }
    }

    answer.trim().to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        fail("usage: pdfllama <pdf-path> <question> [model-path]");
    }

    let pdf_path = &args[1];
    let question = &args[2];
    let model_path = args
        .get(3)
        .cloned()
        .or_else(|| std::env::var("PDFLLAMA_MODEL").ok())
        .unwrap_or_else(|| "models/llama-3.gguf".to_string());

    if !Path::new(pdf_path).is_file() {
        fail(&format!("pdf not found: {}", pdf_path));
    }
    if !Path::new(&model_path).is_file() {
        fail(&format!("gguf model not found: {}", model_path));
    }

    let raw = extract_text(pdf_path);
    let document = clean(&raw);
    if document.is_empty() {
        fail("no extractable text in this pdf");
    }

    let sent = truncate(&document);
    let answer = generate(&model_path, &build_prompt(&sent, question));

    println!(
        "{{\"ok\":true,\"chars_extracted\":{},\"chars_sent\":{},\"truncated\":{},\"answer\":\"{}\"}}",
        document.chars().count(),
        sent.chars().count(),
        document.chars().count() > sent.chars().count(),
        escape_json(&answer)
    );
}
