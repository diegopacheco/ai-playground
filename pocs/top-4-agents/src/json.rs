pub fn get_str(s: &str, key: &str) -> Option<String> {
    let after = value_slice(s, key)?;
    let bytes = after.as_bytes();
    if bytes.first() != Some(&b'"') {
        return None;
    }
    let body = &after[1..];
    let mut out = String::new();
    let mut chars = body.chars();
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => {
                if let Some(n) = chars.next() {
                    match n {
                        'n' => out.push('\n'),
                        't' => out.push('\t'),
                        'r' => out.push('\r'),
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        other => out.push(other),
                    }
                }
            }
            '"' => break,
            _ => out.push(ch),
        }
    }
    Some(out)
}

pub fn get_i64(s: &str, key: &str) -> Option<i64> {
    let after = value_slice(s, key)?;
    let bytes = after.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let mut j = 0;
    if bytes[0] == b'-' {
        j = 1;
    }
    let mut end = 0;
    while j < bytes.len() && bytes[j].is_ascii_digit() {
        j += 1;
        end = j;
    }
    if end == 0 {
        return None;
    }
    after[..end].parse::<i64>().ok()
}

fn value_slice<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{}\"", key);
    let i = s.find(&pat)?;
    let rest = &s[i + pat.len()..];
    let c = rest.find(':')?;
    Some(rest[c + 1..].trim_start())
}
