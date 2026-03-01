pub fn validate_username(username: &str) -> Result<(), String> {
    let trimmed = username.trim();
    if trimmed.is_empty() {
        return Err("Username cannot be empty".to_string());
    }
    if trimmed.chars().count() > 30 {
        return Err("Username must be 30 characters or less".to_string());
    }
    if !trimmed.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err("Username can only contain ASCII letters, numbers, and underscores".to_string());
    }
    Ok(())
}

pub fn validate_password(password: &str) -> Result<(), String> {
    if password.is_empty() {
        return Err("Password cannot be empty".to_string());
    }
    if password.chars().count() > 128 {
        return Err("Password must be 128 characters or less".to_string());
    }
    Ok(())
}

pub fn validate_display_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Display name cannot be empty".to_string());
    }
    if name.chars().count() > 50 {
        return Err("Display name must be 50 characters or less".to_string());
    }
    if name.chars().any(|c| c.is_control()) {
        return Err("Display name cannot contain control characters".to_string());
    }
    Ok(())
}

pub fn validate_bio(bio: &str) -> Result<(), String> {
    if bio.chars().count() > 200 {
        return Err("Bio must be 200 characters or less".to_string());
    }
    if bio.chars().any(|c| c.is_control()) {
        return Err("Bio cannot contain control characters".to_string());
    }
    Ok(())
}

pub fn validate_post_content(content: &str) -> Result<(), String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err("Post content cannot be empty".to_string());
    }
    if content.chars().count() > 280 {
        return Err("Post content must be 280 characters or less".to_string());
    }
    Ok(())
}

pub fn validate_image_type(content_type: &str) -> Result<(), String> {
    match content_type {
        "image/jpeg" | "image/png" => Ok(()),
        _ => Err("Image must be JPEG or PNG".to_string()),
    }
}

pub fn validate_image_size(size: usize) -> Result<(), String> {
    if size > 5 * 1024 * 1024 {
        return Err("Image must be 5MB or less".to_string());
    }
    Ok(())
}

pub fn escape_search_term(term: &str) -> String {
    term.replace('\'', "''")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

pub fn pagination_offset(page: i64, limit: i64) -> i64 {
    (page - 1) * limit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_username() {
        assert!(validate_username("alice").is_ok());
        assert!(validate_username("Bob_123").is_ok());
        assert!(validate_username("a").is_ok());
        assert!(validate_username(&"a".repeat(30)).is_ok());
    }

    #[test]
    fn test_invalid_username_empty() {
        assert!(validate_username("").is_err());
    }

    #[test]
    fn test_invalid_username_too_long() {
        assert!(validate_username(&"a".repeat(31)).is_err());
    }

    #[test]
    fn test_invalid_username_special_chars() {
        assert!(validate_username("user name").is_err());
        assert!(validate_username("user@name").is_err());
        assert!(validate_username("user.name").is_err());
        assert!(validate_username("用户").is_err());
    }

    #[test]
    fn test_invalid_username_whitespace_only() {
        assert!(validate_username("   ").is_err());
    }

    #[test]
    fn test_valid_password() {
        assert!(validate_password("a").is_ok());
        assert!(validate_password(&"x".repeat(128)).is_ok());
    }

    #[test]
    fn test_invalid_password_empty() {
        assert!(validate_password("").is_err());
    }

    #[test]
    fn test_invalid_password_too_long() {
        assert!(validate_password(&"x".repeat(129)).is_err());
    }

    #[test]
    fn test_valid_display_name() {
        assert!(validate_display_name("Alice").is_ok());
        assert!(validate_display_name(&"a".repeat(50)).is_ok());
    }

    #[test]
    fn test_invalid_display_name_empty() {
        assert!(validate_display_name("").is_err());
    }

    #[test]
    fn test_invalid_display_name_too_long() {
        assert!(validate_display_name(&"a".repeat(51)).is_err());
    }

    #[test]
    fn test_invalid_display_name_control_chars() {
        assert!(validate_display_name("hello\x00world").is_err());
        assert!(validate_display_name("\n\n").is_err());
    }

    #[test]
    fn test_valid_bio() {
        assert!(validate_bio("").is_ok());
        assert!(validate_bio("Hello world").is_ok());
        assert!(validate_bio(&"a".repeat(200)).is_ok());
    }

    #[test]
    fn test_invalid_bio_too_long() {
        assert!(validate_bio(&"a".repeat(201)).is_err());
    }

    #[test]
    fn test_valid_post_content() {
        assert!(validate_post_content("Hello").is_ok());
        assert!(validate_post_content(&"a".repeat(280)).is_ok());
    }

    #[test]
    fn test_invalid_post_content_empty() {
        assert!(validate_post_content("").is_err());
    }

    #[test]
    fn test_invalid_post_content_whitespace_only() {
        assert!(validate_post_content("   ").is_err());
    }

    #[test]
    fn test_invalid_post_content_too_long() {
        assert!(validate_post_content(&"a".repeat(281)).is_err());
    }

    #[test]
    fn test_valid_image_types() {
        assert!(validate_image_type("image/jpeg").is_ok());
        assert!(validate_image_type("image/png").is_ok());
    }

    #[test]
    fn test_invalid_image_types() {
        assert!(validate_image_type("text/plain").is_err());
        assert!(validate_image_type("image/gif").is_err());
        assert!(validate_image_type("application/pdf").is_err());
    }

    #[test]
    fn test_valid_image_size() {
        assert!(validate_image_size(0).is_ok());
        assert!(validate_image_size(5 * 1024 * 1024).is_ok());
    }

    #[test]
    fn test_invalid_image_size() {
        assert!(validate_image_size(5 * 1024 * 1024 + 1).is_err());
    }

    #[test]
    fn test_escape_search_term() {
        assert_eq!(escape_search_term("hello"), "hello");
        assert_eq!(escape_search_term("hello%world"), "hello\\%world");
        assert_eq!(escape_search_term("hello_world"), "hello\\_world");
        assert_eq!(escape_search_term("it's"), "it''s");
    }

    #[test]
    fn test_pagination_offset() {
        assert_eq!(pagination_offset(1, 20), 0);
        assert_eq!(pagination_offset(2, 20), 20);
        assert_eq!(pagination_offset(3, 10), 20);
    }
}
