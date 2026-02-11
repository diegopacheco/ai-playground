
#[cfg(test)]
mod tests {
    use crate::models::CreateCommentRequest;
    use validator::Validate;

    #[test]
    fn create_comment_validates_minimum_length() {
        let request = CreateCommentRequest {
            content: "".to_string(),
        };
        assert!(request.validate().is_err());

        let request = CreateCommentRequest {
            content: "x".to_string(),
        };
        assert!(request.validate().is_ok());
    }

    #[test]
    fn create_comment_validates_maximum_length() {
        let request = CreateCommentRequest {
            content: "a".repeat(280),
        };
        assert!(request.validate().is_ok());

        let request = CreateCommentRequest {
            content: "a".repeat(281),
        };
        assert!(request.validate().is_err());
    }

    #[test]
    fn create_comment_validates_exact_280_chars() {
        let content = "a".repeat(280);
        let request = CreateCommentRequest { content };
        assert!(request.validate().is_ok());
    }

    #[test]
    fn create_comment_validates_valid_content() {
        let request = CreateCommentRequest {
            content: "This is a valid comment".to_string(),
        };
        assert!(request.validate().is_ok());
    }
}
