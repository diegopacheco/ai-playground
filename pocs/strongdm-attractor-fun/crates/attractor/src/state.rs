use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct State {
    scopes: Vec<HashMap<String, String>>,
}

impl State {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    pub fn set(&mut self, key: &str, value: String) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(key.to_string(), value);
        }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(key) {
                return Some(val.as_str());
            }
        }
        None
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    pub fn resolve_template(&self, template: &str) -> String {
        let mut result = template.to_string();
        for scope in self.scopes.iter().rev() {
            for (key, value) in scope {
                result = result.replace(&format!("${{{}}}", key), value);
                result = result.replace(&format!("${}", key), value);
            }
        }
        result
    }

    pub fn all_vars(&self) -> HashMap<String, String> {
        let mut merged = HashMap::new();
        for scope in &self.scopes {
            for (k, v) in scope {
                merged.insert(k.clone(), v.clone());
            }
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_state() {
        let mut s = State::new();
        s.set("name", "test".into());
        assert_eq!(s.get("name"), Some("test"));
    }

    #[test]
    fn test_scoped_state() {
        let mut s = State::new();
        s.set("a", "1".into());
        s.push_scope();
        s.set("b", "2".into());
        assert_eq!(s.get("a"), Some("1"));
        assert_eq!(s.get("b"), Some("2"));
        s.pop_scope();
        assert_eq!(s.get("a"), Some("1"));
        assert_eq!(s.get("b"), None);
    }

    #[test]
    fn test_template_resolution() {
        let mut s = State::new();
        s.set("name", "world".into());
        assert_eq!(s.resolve_template("hello ${name}"), "hello world");
        assert_eq!(s.resolve_template("hello $name"), "hello world");
    }
}
