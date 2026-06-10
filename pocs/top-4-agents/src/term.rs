use crate::render::{Color, Line};
use std::io::{Read, Write};
use std::process::Command;

pub struct Term {
    orig: String,
}

impl Term {
    pub fn enter() -> Term {
        let orig = Command::new("stty")
            .arg("-g")
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        let _ = Command::new("stty")
            .args(["-echo", "-icanon", "-isig", "min", "0", "time", "0"])
            .status();
        print!("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H");
        let _ = std::io::stdout().flush();
        Term { orig }
    }

    pub fn size(&self) -> (usize, usize) {
        let out = Command::new("stty").arg("size").output();
        if let Ok(o) = out {
            let s = String::from_utf8_lossy(&o.stdout);
            let nums: Vec<usize> = s.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if nums.len() == 2 && nums[0] > 0 && nums[1] > 0 {
                return (nums[1], nums[0]);
            }
        }
        (116, 30)
    }

    pub fn draw(&self, frame: &[Line], width: usize) {
        let mut buf = String::from("\x1b[H");
        for line in frame {
            buf.push_str(&ansi_line(line, width));
            buf.push_str("\x1b[K\r\n");
        }
        buf.push_str("\x1b[J");
        let mut out = std::io::stdout();
        let _ = out.write_all(buf.as_bytes());
        let _ = out.flush();
    }
}

impl Drop for Term {
    fn drop(&mut self) {
        print!("\x1b[?25h\x1b[?1049l");
        let _ = std::io::stdout().flush();
        if !self.orig.is_empty() {
            let _ = Command::new("stty").arg(&self.orig).status();
        }
    }
}

pub fn poll_keys() -> Vec<u8> {
    let mut buf = [0u8; 16];
    match std::io::stdin().read(&mut buf) {
        Ok(n) if n > 0 => buf[..n].to_vec(),
        _ => Vec::new(),
    }
}

pub fn ansi_line(line: &Line, max: usize) -> String {
    let mut out = String::new();
    let mut count = 0usize;
    for span in line {
        if count >= max {
            break;
        }
        out.push_str(&style_open(span));
        for ch in span.text.chars() {
            if count >= max {
                break;
            }
            out.push(ch);
            count += 1;
        }
        out.push_str("\x1b[0m");
    }
    out
}

fn style_open(span: &crate::render::Span) -> String {
    let mut s = String::new();
    if span.bold {
        s.push_str("\x1b[1m");
    }
    if span.dim {
        s.push_str("\x1b[2m");
    }
    if let Color::Rgb(r, g, b) = span.fg {
        s.push_str(&format!("\x1b[38;2;{};{};{}m", r, g, b));
    }
    s
}
