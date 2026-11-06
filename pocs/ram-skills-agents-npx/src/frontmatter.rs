#[derive(Debug, Default, PartialEq, Eq)]
pub struct Meta {
    pub name: Option<String>,
    pub description: Option<String>,
}

pub fn parse(text: &str) -> Meta {
    let mut lines = text.lines();
    let mut meta = Meta::default();
    if lines.next().map(str::trim) != Some("---") {
        meta.description = first_sentence(text);
        return meta;
    }
    for line in lines.by_ref() {
        if line.trim() == "---" {
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            let value = clean(value);
            if value.is_empty() {
                continue;
            }
            match key.trim() {
                "name" => meta.name = Some(value),
                "description" => meta.description = Some(value),
                _ => {}
            }
        }
    }
    if meta.description.is_none() {
        meta.description = first_sentence(&lines.collect::<Vec<_>>().join("\n"));
    }
    meta
}

fn clean(value: &str) -> String {
    value.trim().trim_matches(|c| c == '"' || c == '\'').trim().to_string()
}

fn first_sentence(body: &str) -> Option<String> {
    body.lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#') && !line.starts_with("---"))
        .map(|line| line.replace('`', ""))
}

pub fn slash_name(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix("# /"))
        .and_then(|rest| rest.split_whitespace().next())
        .map(str::to_string)
}
