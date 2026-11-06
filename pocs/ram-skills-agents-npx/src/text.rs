pub fn char_width(c: char) -> usize {
    match c as u32 {
        0x200D | 0xFE00..=0xFE0F | 0x0300..=0x036F => 0,
        0x1100..=0x115F | 0x2600..=0x27BF | 0x2E80..=0xA4CF | 0xAC00..=0xD7A3 | 0xF900..=0xFAFF | 0xFF00..=0xFF60 | 0x1F000..=0x1FAFF => 2,
        _ => 1,
    }
}

fn clean(text: &str) -> String {
    text.replace('\t', "    ").chars().filter(|c| char_width(*c) > 0 || *c == ' ').filter(|c| !c.is_control()).collect()
}

pub fn width(text: &str) -> usize {
    text.chars().map(char_width).sum()
}

pub fn truncate(text: &str, max: usize) -> String {
    let clean = clean(text);
    if width(&clean) <= max {
        return clean;
    }
    let mut used = 0;
    let mut cut = String::new();
    for c in clean.chars() {
        let w = char_width(c);
        if used + w + 1 > max {
            break;
        }
        used += w;
        cut.push(c);
    }
    cut.push('…');
    cut
}

pub fn wrap(text: &str, max: usize) -> Vec<String> {
    let clean = clean(text);
    let max = max.max(8);
    if width(&clean) <= max {
        return vec![clean];
    }
    let indent: String = clean.chars().take_while(|c| *c == ' ').collect();
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in clean.split(' ').filter(|w| !w.is_empty()) {
        let prefix = if lines.is_empty() && line.is_empty() { "" } else { indent.as_str() };
        let candidate = if line.is_empty() { format!("{prefix}{word}") } else { format!("{line} {word}") };
        if width(&candidate) <= max {
            line = candidate;
            continue;
        }
        if !line.is_empty() {
            lines.push(std::mem::take(&mut line));
        }
        let mut chunk = indent.clone();
        for c in word.chars() {
            if width(&chunk) + char_width(c) > max {
                lines.push(std::mem::replace(&mut chunk, indent.clone()));
            }
            chunk.push(c);
        }
        line = chunk;
    }
    if !line.is_empty() {
        lines.push(line);
    }
    if !indent.is_empty()
        && let Some(first) = lines.first_mut()
        && !first.starts_with(&indent)
    {
        first.insert_str(0, &indent);
    }
    lines
}
