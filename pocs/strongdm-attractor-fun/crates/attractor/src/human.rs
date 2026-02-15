use std::io::{self, BufRead, Write};

pub fn prompt_human(prompt: &str) -> Result<String, String> {
    print!("{}: ", prompt);
    io::stdout()
        .flush()
        .map_err(|e| format!("flush error: {}", e))?;
    let mut input = String::new();
    io::stdin()
        .lock()
        .read_line(&mut input)
        .map_err(|e| format!("read error: {}", e))?;
    Ok(input.trim().to_string())
}
