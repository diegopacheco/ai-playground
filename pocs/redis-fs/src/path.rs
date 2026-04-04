pub fn resolve(cwd: &str, path: &str) -> String {
    if path.starts_with('/') {
        normalize(path)
    } else {
        let full = if cwd == "/" {
            format!("/{}", path)
        } else {
            format!("{}/{}", cwd, path)
        };
        normalize(&full)
    }
}

pub fn normalize(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            _ => parts.push(part),
        }
    }
    if parts.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", parts.join("/"))
    }
}

pub fn parent(path: &str) -> String {
    if path == "/" {
        return "/".to_string();
    }
    match path.rfind('/') {
        Some(0) => "/".to_string(),
        Some(i) => path[..i].to_string(),
        None => "/".to_string(),
    }
}

pub fn basename(path: &str) -> String {
    match path.rfind('/') {
        Some(i) => path[i + 1..].to_string(),
        None => path.to_string(),
    }
}
