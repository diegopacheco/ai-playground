use crate::state::State;

#[derive(Debug, Clone)]
pub enum CondExpr {
    Eq(String, String),
    Neq(String, String),
    Contains(String, String),
    And(Box<CondExpr>, Box<CondExpr>),
    Or(Box<CondExpr>, Box<CondExpr>),
    Not(Box<CondExpr>),
    Literal(bool),
}

impl CondExpr {
    pub fn evaluate(&self, state: &State) -> bool {
        match self {
            CondExpr::Eq(var, val) => {
                let resolved = state.get(var.trim_start_matches('$')).unwrap_or("");
                resolved == val
            }
            CondExpr::Neq(var, val) => {
                let resolved = state.get(var.trim_start_matches('$')).unwrap_or("");
                resolved != val
            }
            CondExpr::Contains(var, val) => {
                let resolved = state.get(var.trim_start_matches('$')).unwrap_or("");
                resolved.contains(val.as_str())
            }
            CondExpr::And(a, b) => a.evaluate(state) && b.evaluate(state),
            CondExpr::Or(a, b) => a.evaluate(state) || b.evaluate(state),
            CondExpr::Not(a) => !a.evaluate(state),
            CondExpr::Literal(v) => *v,
        }
    }
}

pub fn parse_condition(expr: &str) -> Result<CondExpr, String> {
    let expr = expr.trim();
    if let Some(pos) = find_binary_op(expr, " or ") {
        let left = parse_condition(&expr[..pos])?;
        let right = parse_condition(&expr[pos + 4..])?;
        return Ok(CondExpr::Or(Box::new(left), Box::new(right)));
    }
    if let Some(pos) = find_binary_op(expr, " and ") {
        let left = parse_condition(&expr[..pos])?;
        let right = parse_condition(&expr[pos + 5..])?;
        return Ok(CondExpr::And(Box::new(left), Box::new(right)));
    }
    if expr.starts_with("not ") {
        let inner = parse_condition(&expr[4..])?;
        return Ok(CondExpr::Not(Box::new(inner)));
    }
    if expr == "true" {
        return Ok(CondExpr::Literal(true));
    }
    if expr == "false" {
        return Ok(CondExpr::Literal(false));
    }
    if let Some(pos) = expr.find("!=") {
        let var = expr[..pos].trim().to_string();
        let val = expr[pos + 2..].trim().trim_matches('"').to_string();
        return Ok(CondExpr::Neq(var, val));
    }
    if let Some(pos) = expr.find("==") {
        let var = expr[..pos].trim().to_string();
        let val = expr[pos + 2..].trim().trim_matches('"').to_string();
        return Ok(CondExpr::Eq(var, val));
    }
    if expr.contains(" contains ") {
        let parts: Vec<&str> = expr.splitn(2, " contains ").collect();
        let var = parts[0].trim().to_string();
        let val = parts[1].trim().trim_matches('"').to_string();
        return Ok(CondExpr::Contains(var, val));
    }
    Err(format!("cannot parse condition: {}", expr))
}

fn find_binary_op(expr: &str, op: &str) -> Option<usize> {
    expr.find(op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::State;

    #[test]
    fn test_eq_condition() {
        let mut state = State::new();
        state.set("status", "ok".into());
        let cond = parse_condition("$status == \"ok\"").unwrap();
        assert!(cond.evaluate(&state));
    }

    #[test]
    fn test_neq_condition() {
        let mut state = State::new();
        state.set("status", "err".into());
        let cond = parse_condition("$status != \"ok\"").unwrap();
        assert!(cond.evaluate(&state));
    }

    #[test]
    fn test_and_condition() {
        let mut state = State::new();
        state.set("a", "1".into());
        state.set("b", "2".into());
        let cond = parse_condition("$a == \"1\" and $b == \"2\"").unwrap();
        assert!(cond.evaluate(&state));
    }

    #[test]
    fn test_or_condition() {
        let mut state = State::new();
        state.set("a", "1".into());
        let cond = parse_condition("$a == \"1\" or $a == \"2\"").unwrap();
        assert!(cond.evaluate(&state));
    }

    #[test]
    fn test_literal() {
        let state = State::new();
        let cond = parse_condition("true").unwrap();
        assert!(cond.evaluate(&state));
    }
}
