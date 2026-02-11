use twitter_clone::middleware::{create_jwt, verify_jwt};

#[test]
fn test_jwt_creation_and_verification() {
    let user_id = 123;
    let secret = "test-secret";

    let token = create_jwt(user_id, secret).expect("Failed to create JWT");

    assert!(!token.is_empty());

    let claims = verify_jwt(&token, secret).expect("Failed to verify JWT");

    assert_eq!(claims.sub, user_id);
}

#[test]
fn test_jwt_verification_with_wrong_secret() {
    let user_id = 123;
    let secret = "test-secret";
    let wrong_secret = "wrong-secret";

    let token = create_jwt(user_id, secret).expect("Failed to create JWT");

    let result = verify_jwt(&token, wrong_secret);

    assert!(result.is_err());
}

#[test]
fn test_jwt_verification_with_invalid_token() {
    let secret = "test-secret";
    let invalid_token = "invalid.token.here";

    let result = verify_jwt(invalid_token, secret);

    assert!(result.is_err());
}
