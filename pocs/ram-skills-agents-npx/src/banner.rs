use crossterm::style::{Color, Stylize};
use std::io::{IsTerminal, Write, stdout};
use std::time::Duration;

const ART: [&str; 6] = [
    "██████╗  █████╗ ███╗   ███╗",
    "██╔══██╗██╔══██╗████╗ ████║",
    "██████╔╝███████║██╔████╔██║",
    "██╔══██╗██╔══██║██║╚██╔╝██║",
    "██║  ██║██║  ██║██║ ╚═╝ ██║",
    "╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝",
];

const FRAMES: usize = 18;

pub fn hue(position: f32) -> Color {
    let h = position.rem_euclid(1.0) * 6.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as u32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    let lift = |v: f32| (90.0 + v * 165.0) as u8;
    Color::Rgb { r: lift(r), g: lift(g), b: lift(b) }
}

pub fn render(shift: f32) -> String {
    let width = ART[0].chars().count() as f32;
    let mut out = String::new();
    for (row, line) in ART.iter().enumerate() {
        out.push_str("   ");
        for (col, ch) in line.chars().enumerate() {
            let position = shift + col as f32 / width * 0.6 + row as f32 * 0.03;
            out.push_str(&ch.to_string().with(hue(position)).bold().to_string());
        }
        out.push_str("\r\n");
    }
    out.push_str(&format!(
        "   {}  {}\r\n",
        "skills · commands · agents · plugins".with(Color::Grey),
        format!("v{}", env!("CARGO_PKG_VERSION")).with(Color::DarkGrey)
    ));
    out
}

pub async fn play() -> std::io::Result<()> {
    let mut out = stdout();
    if !out.is_terminal() || !crate::style::enabled() {
        for line in ART {
            writeln!(out, "   {line}")?;
        }
        writeln!(out, "   skills · commands · agents · plugins  v{}\n", env!("CARGO_PKG_VERSION"))?;
        return Ok(());
    }
    let height = ART.len() + 1;
    for frame in 0..FRAMES {
        if frame > 0 {
            write!(out, "\x1b[{height}A")?;
        }
        write!(out, "{}", render(frame as f32 / FRAMES as f32))?;
        out.flush()?;
        tokio::time::sleep(Duration::from_millis(28)).await;
    }
    writeln!(out)?;
    Ok(())
}
