pub fn quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other if (other as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", other as u32)),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

pub fn object(fields: &[(&str, String)]) -> String {
    let body: Vec<String> = fields.iter().map(|(name, value)| format!("{}:{value}", quote(name))).collect();
    format!("{{{}}}", body.join(","))
}

pub fn list(values: &[String]) -> String {
    format!("[{}]", values.join(","))
}

pub fn score(value: f32) -> String {
    format!("{value:.4}")
}

pub fn optional_score(value: Option<f32>) -> String {
    value.map(score).unwrap_or_else(|| "null".into())
}

pub fn optional_text(value: Option<&str>) -> String {
    value.map(quote).unwrap_or_else(|| "null".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_sdk::json::Json;

    #[test]
    fn claude_answers_with_quotes_and_newlines_survive_a_round_trip_to_the_ui() {
        let answer = "Use \"cargo\"\n\tthen C:\\path \u{1}";
        let body = object(&[("answer", quote(answer))]);
        assert_eq!(Json::object(&body).get("answer").and_then(Json::text), Some(answer));
    }

    #[test]
    fn missing_values_are_json_null_so_the_ui_can_tell_absent_from_zero() {
        assert_eq!(optional_score(None), "null");
        assert_eq!(optional_score(Some(0.0)), "0.0000");
        assert_eq!(optional_text(None), "null");
    }
}
