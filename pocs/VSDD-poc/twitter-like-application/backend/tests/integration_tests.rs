use actix_web::{test, web, App};
use twitter_backend::{create_app_state, configure_routes};

fn test_state() -> web::Data<twitter_backend::db::AppState> {
    create_app_state(":memory:")
}

#[actix_rt::test]
async fn test_register_success() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::post()
        .uri("/api/auth/register")
        .set_json(serde_json::json!({
            "username": "testuser",
            "password": "pass123",
            "display_name": "Test User"
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 201);
}

#[actix_rt::test]
async fn test_register_empty_username() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::post()
        .uri("/api/auth/register")
        .set_json(serde_json::json!({
            "username": "",
            "password": "pass123",
            "display_name": "Test"
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 400);
}

#[actix_rt::test]
async fn test_register_username_too_long() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::post()
        .uri("/api/auth/register")
        .set_json(serde_json::json!({
            "username": "a".repeat(31),
            "password": "pass123",
            "display_name": "Test"
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 400);
}

#[actix_rt::test]
async fn test_register_invalid_username_chars() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::post()
        .uri("/api/auth/register")
        .set_json(serde_json::json!({
            "username": "user@name",
            "password": "pass123",
            "display_name": "Test"
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 400);
}

#[actix_rt::test]
async fn test_register_duplicate_username() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let body = serde_json::json!({
        "username": "testuser",
        "password": "pass123",
        "display_name": "Test"
    });
    let req = test::TestRequest::post()
        .uri("/api/auth/register")
        .set_json(&body)
        .to_request();
    test::call_service(&app, req).await;
    let req2 = test::TestRequest::post()
        .uri("/api/auth/register")
        .set_json(&body)
        .to_request();
    let resp = test::call_service(&app, req2).await;
    assert_eq!(resp.status(), 409);
}

#[actix_rt::test]
async fn test_register_duplicate_username_case_insensitive() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::post()
        .uri("/api/auth/register")
        .set_json(serde_json::json!({
            "username": "Alice",
            "password": "pass123",
            "display_name": "A"
        }))
        .to_request();
    test::call_service(&app, req).await;
    let req2 = test::TestRequest::post()
        .uri("/api/auth/register")
        .set_json(serde_json::json!({
            "username": "alice",
            "password": "pass123",
            "display_name": "B"
        }))
        .to_request();
    let resp = test::call_service(&app, req2).await;
    assert_eq!(resp.status(), 409);
}

#[actix_rt::test]
async fn test_login_success() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::post()
        .uri("/api/auth/register")
        .set_json(serde_json::json!({
            "username": "testuser",
            "password": "pass123",
            "display_name": "Test"
        }))
        .to_request();
    test::call_service(&app, req).await;
    let req2 = test::TestRequest::post()
        .uri("/api/auth/login")
        .set_json(serde_json::json!({
            "username": "testuser",
            "password": "pass123"
        }))
        .to_request();
    let resp = test::call_service(&app, req2).await;
    assert_eq!(resp.status(), 200);
}

#[actix_rt::test]
async fn test_login_wrong_password() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::post()
        .uri("/api/auth/register")
        .set_json(serde_json::json!({
            "username": "testuser",
            "password": "pass123",
            "display_name": "Test"
        }))
        .to_request();
    test::call_service(&app, req).await;
    let req2 = test::TestRequest::post()
        .uri("/api/auth/login")
        .set_json(serde_json::json!({
            "username": "testuser",
            "password": "wrong"
        }))
        .to_request();
    let resp = test::call_service(&app, req2).await;
    assert_eq!(resp.status(), 401);
}

#[actix_rt::test]
async fn test_login_default_admin() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::post()
        .uri("/api/auth/login")
        .set_json(serde_json::json!({
            "username": "admin",
            "password": "admin"
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
}

#[actix_rt::test]
async fn test_me_unauthenticated() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::get()
        .uri("/api/auth/me")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 401);
}

#[actix_rt::test]
async fn test_create_post_unauthenticated() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::post()
        .uri("/api/posts")
        .set_json(serde_json::json!({"content": "hello"}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 401);
}

#[actix_rt::test]
async fn test_get_post_not_found() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::get()
        .uri("/api/posts/9999")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);
}

#[actix_rt::test]
async fn test_like_unauthenticated() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::post()
        .uri("/api/posts/1/like")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 401);
}

#[actix_rt::test]
async fn test_follow_self() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::post()
        .uri("/api/auth/login")
        .set_json(serde_json::json!({"username": "admin", "password": "admin"}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    let cookies = resp.response().headers().get_all("set-cookie")
        .map(|v| v.to_str().unwrap().to_string())
        .collect::<Vec<_>>();
    let cookie = cookies.iter().find(|c| c.starts_with("session_id=")).unwrap().clone();
    let session_val = cookie.split('=').nth(1).unwrap().split(';').next().unwrap();
    let req2 = test::TestRequest::post()
        .uri("/api/users/1/follow")
        .insert_header(("cookie", format!("session_id={}", session_val)))
        .to_request();
    let resp2 = test::call_service(&app, req2).await;
    assert_eq!(resp2.status(), 400);
}

#[actix_rt::test]
async fn test_follow_nonexistent_user() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::post()
        .uri("/api/auth/login")
        .set_json(serde_json::json!({"username": "admin", "password": "admin"}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    let cookies = resp.response().headers().get_all("set-cookie")
        .map(|v| v.to_str().unwrap().to_string())
        .collect::<Vec<_>>();
    let cookie = cookies.iter().find(|c| c.starts_with("session_id=")).unwrap().clone();
    let session_val = cookie.split('=').nth(1).unwrap().split(';').next().unwrap();
    let req2 = test::TestRequest::post()
        .uri("/api/users/9999/follow")
        .insert_header(("cookie", format!("session_id={}", session_val)))
        .to_request();
    let resp2 = test::call_service(&app, req2).await;
    assert_eq!(resp2.status(), 404);
}

#[actix_rt::test]
async fn test_timeline_unauthenticated() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::get()
        .uri("/api/timeline")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 401);
}

#[actix_rt::test]
async fn test_search_empty_query() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::get()
        .uri("/api/search?q=&type=posts")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 400);
}

#[actix_rt::test]
async fn test_search_invalid_type() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::get()
        .uri("/api/search?q=hello&type=invalid")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 400);
}

#[actix_rt::test]
async fn test_get_user_profile_not_found() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::get()
        .uri("/api/users/9999")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);
}

#[actix_rt::test]
async fn test_update_profile_unauthenticated() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::put()
        .uri("/api/users/1")
        .set_json(serde_json::json!({"display_name": "New", "bio": "hi"}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 401);
}

#[actix_rt::test]
async fn test_hot_topics_empty() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::get()
        .uri("/api/hot")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = test::read_body_json(resp).await;
    assert_eq!(body["posts"].as_array().unwrap().len(), 0);
}

#[actix_rt::test]
async fn test_get_admin_profile() {
    let state = test_state();
    let app = test::init_service(
        App::new().app_data(state.clone()).configure(configure_routes)
    ).await;
    let req = test::TestRequest::get()
        .uri("/api/users/1")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = test::read_body_json(resp).await;
    assert_eq!(body["username"], "admin");
}
