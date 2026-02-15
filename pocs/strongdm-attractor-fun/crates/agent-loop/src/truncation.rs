const MAX_OUTPUT_CHARS: usize = 50000;
const HEAD_CHARS: usize = 20000;
const TAIL_CHARS: usize = 20000;

pub fn truncate_output(output: &str) -> String {
    if output.len() <= MAX_OUTPUT_CHARS {
        return output.to_string();
    }
    let head = &output[..HEAD_CHARS];
    let tail = &output[output.len() - TAIL_CHARS..];
    let omitted = output.len() - HEAD_CHARS - TAIL_CHARS;
    format!(
        "{}\n\n... ({} characters omitted) ...\n\n{}",
        head, omitted, tail
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_output_unchanged() {
        let s = "hello world";
        assert_eq!(truncate_output(s), s);
    }

    #[test]
    fn test_long_output_truncated() {
        let s = "x".repeat(60000);
        let result = truncate_output(&s);
        assert!(result.len() < s.len());
        assert!(result.contains("characters omitted"));
    }
}
