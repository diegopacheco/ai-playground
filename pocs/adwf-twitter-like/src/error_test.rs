
#[cfg(test)]
mod tests {
    use crate::error::AppError;
    use axum::{http::StatusCode, response::IntoResponse};

    #[test]
    fn authentication_error_returns_401() {
        let error = AppError::Authentication("Invalid credentials".to_string());
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn authorization_error_returns_403() {
        let error = AppError::Authorization("Access denied".to_string());
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[test]
    fn not_found_error_returns_404() {
        let error = AppError::NotFound("Resource not found".to_string());
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn bad_request_error_returns_400() {
        let error = AppError::BadRequest("Invalid input".to_string());
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn internal_error_returns_500() {
        let error = AppError::Internal(anyhow::anyhow!("Something went wrong"));
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn database_error_returns_500() {
        let error = AppError::Database(sqlx::Error::RowNotFound);
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn authentication_error_has_correct_message() {
        let msg = "Invalid token".to_string();
        let error = AppError::Authentication(msg.clone());
        assert_eq!(error.to_string(), format!("Authentication error: {}", msg));
    }

    #[test]
    fn authorization_error_has_correct_message() {
        let msg = "No permission".to_string();
        let error = AppError::Authorization(msg.clone());
        assert_eq!(error.to_string(), format!("Authorization error: {}", msg));
    }

    #[test]
    fn not_found_error_has_correct_message() {
        let msg = "User not found".to_string();
        let error = AppError::NotFound(msg.clone());
        assert_eq!(error.to_string(), format!("Not found: {}", msg));
    }

    #[test]
    fn bad_request_error_has_correct_message() {
        let msg = "Invalid data".to_string();
        let error = AppError::BadRequest(msg.clone());
        assert_eq!(error.to_string(), format!("Bad request: {}", msg));
    }
}
