#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    Text(String),
    List(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(input: &str) -> Option<Json> {
        let characters: Vec<char> = input.chars().collect();
        let mut cursor = 0;
        parse_value(&characters, &mut cursor)
    }

    pub fn object(input: &str) -> Json {
        let start = match input.find('{') {
            Some(position) => position,
            None => return Json::Object(Vec::new()),
        };
        let end = match input.rfind('}') {
            Some(position) if position > start => position,
            _ => return Json::Object(Vec::new()),
        };
        match Json::parse(&input[start..=end]) {
            Some(value @ Json::Object(_)) => value,
            _ => Json::Object(Vec::new()),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(entries) => entries.iter().find(|(name, _)| name == key).map(|(_, value)| value),
            _ => None,
        }
    }

    pub fn number(&self) -> Option<f64> {
        match self {
            Json::Number(value) => Some(*value),
            _ => None,
        }
    }

    pub fn text(&self) -> Option<&str> {
        match self {
            Json::Text(value) => Some(value),
            _ => None,
        }
    }

    pub fn list(&self) -> Option<&[Json]> {
        match self {
            Json::List(values) => Some(values),
            _ => None,
        }
    }

    pub fn is_true(&self) -> bool {
        matches!(self, Json::Bool(true))
    }
}

fn skip_spaces(input: &[char], cursor: &mut usize) {
    while *cursor < input.len() && input[*cursor].is_whitespace() {
        *cursor += 1;
    }
}

fn parse_value(input: &[char], cursor: &mut usize) -> Option<Json> {
    skip_spaces(input, cursor);
    match *input.get(*cursor)? {
        '{' => parse_object(input, cursor),
        '[' => parse_list(input, cursor),
        '"' => parse_text(input, cursor).map(Json::Text),
        't' => parse_literal(input, cursor, "true", Json::Bool(true)),
        'f' => parse_literal(input, cursor, "false", Json::Bool(false)),
        'n' => parse_literal(input, cursor, "null", Json::Null),
        _ => parse_number(input, cursor),
    }
}

fn parse_literal(input: &[char], cursor: &mut usize, word: &str, value: Json) -> Option<Json> {
    for expected in word.chars() {
        if *input.get(*cursor)? != expected {
            return None;
        }
        *cursor += 1;
    }
    Some(value)
}

fn parse_number(input: &[char], cursor: &mut usize) -> Option<Json> {
    let start = *cursor;
    while *cursor < input.len() && matches!(input[*cursor], '0'..='9' | '-' | '+' | '.' | 'e' | 'E') {
        *cursor += 1;
    }
    if start == *cursor {
        return None;
    }
    input[start..*cursor].iter().collect::<String>().parse::<f64>().ok().map(Json::Number)
}

fn parse_text(input: &[char], cursor: &mut usize) -> Option<String> {
    if *input.get(*cursor)? != '"' {
        return None;
    }
    *cursor += 1;
    let mut value = String::new();
    loop {
        let current = *input.get(*cursor)?;
        *cursor += 1;
        match current {
            '"' => return Some(value),
            '\\' => {
                let escape = *input.get(*cursor)?;
                *cursor += 1;
                match escape {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    'b' => value.push('\u{8}'),
                    'f' => value.push('\u{c}'),
                    'u' => value.push(parse_escape(input, cursor)?),
                    other => value.push(other),
                }
            }
            other => value.push(other),
        }
    }
}

fn parse_escape(input: &[char], cursor: &mut usize) -> Option<char> {
    let mut code = 0u32;
    for _ in 0..4 {
        code = code * 16 + input.get(*cursor)?.to_digit(16)?;
        *cursor += 1;
    }
    if (0xD800..0xDC00).contains(&code) && input.get(*cursor) == Some(&'\\') && input.get(*cursor + 1) == Some(&'u') {
        *cursor += 2;
        let mut low = 0u32;
        for _ in 0..4 {
            low = low * 16 + input.get(*cursor)?.to_digit(16)?;
            *cursor += 1;
        }
        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
    }
    char::from_u32(code).or(Some('\u{fffd}'))
}

fn parse_list(input: &[char], cursor: &mut usize) -> Option<Json> {
    *cursor += 1;
    let mut values = Vec::new();
    loop {
        skip_spaces(input, cursor);
        match *input.get(*cursor)? {
            ']' => {
                *cursor += 1;
                return Some(Json::List(values));
            }
            ',' => *cursor += 1,
            _ => values.push(parse_value(input, cursor)?),
        }
    }
}

fn parse_object(input: &[char], cursor: &mut usize) -> Option<Json> {
    *cursor += 1;
    let mut entries = Vec::new();
    loop {
        skip_spaces(input, cursor);
        match *input.get(*cursor)? {
            '}' => {
                *cursor += 1;
                return Some(Json::Object(entries));
            }
            ',' => *cursor += 1,
            '"' => {
                let key = parse_text(input, cursor)?;
                skip_spaces(input, cursor);
                if *input.get(*cursor)? != ':' {
                    return None;
                }
                *cursor += 1;
                entries.push((key, parse_value(input, cursor)?));
            }
            _ => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_values_are_reachable_by_key() {
        let value = Json::object("banner {\"a\":{\"b\":[1,2.5,true,null,\"x\\\"y\"]}} trailing");
        let list = value.get("a").unwrap().get("b").unwrap().list().unwrap();
        assert_eq!(list[1].number(), Some(2.5));
        assert!(list[2].is_true());
        assert_eq!(list[4].text(), Some("x\"y"));
    }

    #[test]
    fn a_broken_document_is_an_empty_object() {
        assert_eq!(Json::object("not json"), Json::Object(Vec::new()));
        assert_eq!(Json::object("{\"a\":"), Json::Object(Vec::new()));
    }

    #[test]
    fn escaped_unicode_is_decoded() {
        assert_eq!(Json::object(r#"{"a":"Aé"}"#).get("a").unwrap().text(), Some("Aé"));
    }
}
