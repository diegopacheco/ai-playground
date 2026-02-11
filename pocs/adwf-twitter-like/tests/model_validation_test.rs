use twitter_clone::models::*;
use validator::Validate;

#[test]
fn test_register_request_validation() {
    let valid_request = RegisterRequest {
        username: "validuser".to_string(),
        email: "valid@email.com".to_string(),
        password: "password123".to_string(),
    };
    assert!(valid_request.validate().is_ok());

    let invalid_username = RegisterRequest {
        username: "ab".to_string(),
        email: "valid@email.com".to_string(),
        password: "password123".to_string(),
    };
    assert!(invalid_username.validate().is_err());

    let invalid_email = RegisterRequest {
        username: "validuser".to_string(),
        email: "invalid-email".to_string(),
        password: "password123".to_string(),
    };
    assert!(invalid_email.validate().is_err());

    let invalid_password = RegisterRequest {
        username: "validuser".to_string(),
        email: "valid@email.com".to_string(),
        password: "short".to_string(),
    };
    assert!(invalid_password.validate().is_err());
}

#[test]
fn test_create_tweet_validation() {
    let valid_tweet = CreateTweetRequest {
        content: "This is a valid tweet".to_string(),
    };
    assert!(valid_tweet.validate().is_ok());

    let empty_tweet = CreateTweetRequest {
        content: "".to_string(),
    };
    assert!(empty_tweet.validate().is_err());

    let long_tweet = CreateTweetRequest {
        content: "a".repeat(281),
    };
    assert!(long_tweet.validate().is_err());
}

#[test]
fn test_create_comment_validation() {
    let valid_comment = CreateCommentRequest {
        content: "This is a valid comment".to_string(),
    };
    assert!(valid_comment.validate().is_ok());

    let empty_comment = CreateCommentRequest {
        content: "".to_string(),
    };
    assert!(empty_comment.validate().is_err());

    let long_comment = CreateCommentRequest {
        content: "a".repeat(281),
    };
    assert!(long_comment.validate().is_err());
}

#[test]
fn test_update_user_validation() {
    let valid_update = UpdateUserRequest {
        display_name: Some("Valid Name".to_string()),
        bio: Some("Valid bio".to_string()),
    };
    assert!(valid_update.validate().is_ok());

    let long_display_name = UpdateUserRequest {
        display_name: Some("a".repeat(101)),
        bio: Some("Valid bio".to_string()),
    };
    assert!(long_display_name.validate().is_err());

    let long_bio = UpdateUserRequest {
        display_name: Some("Valid Name".to_string()),
        bio: Some("a".repeat(501)),
    };
    assert!(long_bio.validate().is_err());
}
