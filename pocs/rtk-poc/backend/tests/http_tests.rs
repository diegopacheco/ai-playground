use actix_cors::Cors;
use actix_web::{App, http::StatusCode, test, web};
use memory_backend::configure;
use memory_backend::scores::ScoreStore;
use serde_json::{Value, json};
use std::sync::Arc;

macro_rules! make_app {
    () => {{
        let store = web::Data::from(Arc::new(ScoreStore::new()));
        App::new()
            .wrap(Cors::permissive())
            .app_data(store)
            .configure(configure)
    }};
}

#[actix_rt::test]
async fn health_returns_ok() {
    let app = test::init_service(make_app!()).await;
    let req = test::TestRequest::get().uri("/api/health").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value = test::read_body_json(resp).await;
    assert_eq!(body["status"], "ok");
}

#[actix_rt::test]
async fn new_game_returns_deck_of_16() {
    let app = test::init_service(make_app!()).await;
    let req = test::TestRequest::post().uri("/api/games").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value = test::read_body_json(resp).await;
    assert_eq!(body["deck"].as_array().unwrap().len(), 16);
}

#[actix_rt::test]
async fn new_game_deck_has_unique_ids() {
    let app = test::init_service(make_app!()).await;
    let req = test::TestRequest::post().uri("/api/games").to_request();
    let body: Value = test::call_and_read_body_json(&app, req).await;
    let ids: std::collections::HashSet<u64> = body["deck"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["id"].as_u64().unwrap())
        .collect();
    assert_eq!(ids.len(), 16);
}

#[actix_rt::test]
async fn scores_empty_initially() {
    let app = test::init_service(make_app!()).await;
    let req = test::TestRequest::get().uri("/api/scores").to_request();
    let body: Value = test::call_and_read_body_json(&app, req).await;
    assert_eq!(body["scores"].as_array().unwrap().len(), 0);
}

#[actix_rt::test]
async fn post_score_returns_created() {
    let app = test::init_service(make_app!()).await;
    let req = test::TestRequest::post()
        .uri("/api/scores")
        .set_json(json!({ "name": "diego", "moves": 12, "seconds": 40 }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::CREATED);
}

#[actix_rt::test]
async fn post_score_persists_and_lists() {
    let app = test::init_service(make_app!()).await;
    let post = test::TestRequest::post()
        .uri("/api/scores")
        .set_json(json!({ "name": "diego", "moves": 12, "seconds": 40 }))
        .to_request();
    test::call_service(&app, post).await;
    let get = test::TestRequest::get().uri("/api/scores").to_request();
    let body: Value = test::call_and_read_body_json(&app, get).await;
    let scores = body["scores"].as_array().unwrap();
    assert_eq!(scores.len(), 1);
    assert_eq!(scores[0]["name"], "diego");
    assert_eq!(scores[0]["moves"], 12);
}

#[actix_rt::test]
async fn scores_returned_sorted_by_moves() {
    let app = test::init_service(make_app!()).await;
    for (name, moves) in [("a", 20u32), ("b", 5), ("c", 12)] {
        let req = test::TestRequest::post()
            .uri("/api/scores")
            .set_json(json!({ "name": name, "moves": moves, "seconds": 30 }))
            .to_request();
        test::call_service(&app, req).await;
    }
    let get = test::TestRequest::get().uri("/api/scores").to_request();
    let body: Value = test::call_and_read_body_json(&app, get).await;
    let scores = body["scores"].as_array().unwrap();
    assert_eq!(scores[0]["name"], "b");
    assert_eq!(scores[1]["name"], "c");
    assert_eq!(scores[2]["name"], "a");
}

#[actix_rt::test]
async fn post_score_rejects_empty_name() {
    let app = test::init_service(make_app!()).await;
    let req = test::TestRequest::post()
        .uri("/api/scores")
        .set_json(json!({ "name": "", "moves": 10, "seconds": 20 }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[actix_rt::test]
async fn post_score_rejects_zero_moves() {
    let app = test::init_service(make_app!()).await;
    let req = test::TestRequest::post()
        .uri("/api/scores")
        .set_json(json!({ "name": "x", "moves": 0, "seconds": 20 }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[actix_rt::test]
async fn post_score_rejects_zero_seconds() {
    let app = test::init_service(make_app!()).await;
    let req = test::TestRequest::post()
        .uri("/api/scores")
        .set_json(json!({ "name": "x", "moves": 10, "seconds": 0 }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[actix_rt::test]
async fn unknown_route_returns_404() {
    let app = test::init_service(make_app!()).await;
    let req = test::TestRequest::get().uri("/api/nope").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[actix_rt::test]
async fn cors_headers_present_on_options() {
    let app = test::init_service(make_app!()).await;
    let req = test::TestRequest::default()
        .method(actix_web::http::Method::OPTIONS)
        .uri("/api/scores")
        .insert_header(("Origin", "http://localhost:8000"))
        .insert_header(("Access-Control-Request-Method", "POST"))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.headers().contains_key("access-control-allow-origin"));
}
