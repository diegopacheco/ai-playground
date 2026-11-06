use crossterm::style::{Color, Stylize};
use std::io::{IsTerminal, Write, stdout};
use std::time::Duration;

const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub async fn spin<T>(message: &str, work: impl Future<Output = T>) -> T {
    if !stdout().is_terminal() {
        println!("   {message}");
        return work.await;
    }
    let text = message.to_string();
    let ticker = tokio::spawn(async move {
        let mut frame = 0usize;
        loop {
            let glyph = FRAMES[frame % FRAMES.len()].with(crate::banner::hue(frame as f32 / 40.0)).bold();
            print!("\r   {glyph} {}", text.as_str().with(Color::Grey));
            let _ = stdout().flush();
            frame += 1;
            tokio::time::sleep(Duration::from_millis(70)).await;
        }
    });
    let result = work.await;
    ticker.abort();
    let _ = ticker.await;
    print!("\r\x1b[2K");
    let _ = stdout().flush();
    result
}
