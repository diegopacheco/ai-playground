
#[cfg(test)]
mod tests {
    use crate::models::CreateTweetRequest;
    use validator::Validate;

    #[test]
    fn create_tweet_validates_minimum_length() {
        let request = CreateTweetRequest {
            content: "".to_string(),
        };
        assert!(request.validate().is_err());

        let request = CreateTweetRequest {
            content: "x".to_string(),
        };
        assert!(request.validate().is_ok());
    }

    #[test]
    fn create_tweet_validates_maximum_length() {
        let request = CreateTweetRequest {
            content: "a".repeat(280),
        };
        assert!(request.validate().is_ok());

        let request = CreateTweetRequest {
            content: "a".repeat(281),
        };
        assert!(request.validate().is_err());
    }

    #[test]
    fn create_tweet_validates_exact_280_chars() {
        let content = "a".repeat(280);
        let request = CreateTweetRequest { content };
        assert!(request.validate().is_ok());
    }

    #[test]
    fn create_tweet_validates_whitespace_only() {
        let request = CreateTweetRequest {
            content: "   ".to_string(),
        };
        assert!(request.validate().is_ok());
    }
}
