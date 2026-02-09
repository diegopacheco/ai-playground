use actix_web::{test, web, App};
use person_crud_backend::*;
use rusqlite::Connection;
use std::sync::Mutex;

fn setup_app_data() -> web::Data<AppState> {
    let conn = Connection::open_in_memory().unwrap();
    init_db(&conn);
    web::Data::new(AppState {
        db: Mutex::new(conn),
    })
}

fn build_app(
    data: web::Data<AppState>,
) -> App<
    impl actix_web::dev::ServiceFactory<
        actix_web::dev::ServiceRequest,
        Config = (),
        Response = actix_web::dev::ServiceResponse<impl actix_web::body::MessageBody>,
        Error = actix_web::Error,
        InitError = (),
    >,
> {
    App::new()
        .app_data(data)
        .route("/persons", web::get().to(get_persons))
        .route("/persons/{id}", web::get().to(get_person))
        .route("/persons", web::post().to(create_person))
        .route("/persons/{id}", web::put().to(update_person))
        .route("/persons/{id}", web::delete().to(delete_person))
        .route("/admin/views", web::get().to(get_admin_views))
        .route("/admin/views/{id}", web::get().to(get_admin_view))
}

#[actix_web::test]
async fn test_create_person_and_list() {
    let data = setup_app_data();
    let app = test::init_service(build_app(data)).await;

    let person = serde_json::json!({
        "name": "Alice",
        "email": "alice@test.com",
        "age": 30
    });
    let req = test::TestRequest::post()
        .uri("/persons")
        .set_json(&person)
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 201);

    let body: Person = test::read_body_json(resp).await;
    assert_eq!(body.name, "Alice");
    assert_eq!(body.email, "alice@test.com");
    assert_eq!(body.age, 30);
    assert!(body.id.is_some());

    let req = test::TestRequest::get().uri("/persons").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);

    let persons: Vec<Person> = test::read_body_json(resp).await;
    assert_eq!(persons.len(), 1);
    assert_eq!(persons[0].name, "Alice");
}

#[actix_web::test]
async fn test_create_then_admin_views_returns_zero() {
    let data = setup_app_data();
    let app = test::init_service(build_app(data)).await;

    let person = serde_json::json!({
        "name": "Bob",
        "email": "bob@test.com",
        "age": 25
    });
    let req = test::TestRequest::post()
        .uri("/persons")
        .set_json(&person)
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 201);
    let created: Person = test::read_body_json(resp).await;
    let person_id = created.id.unwrap();

    let req = test::TestRequest::get()
        .uri(&format!("/admin/views/{}", person_id))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);

    let view: PersonView = test::read_body_json(resp).await;
    assert_eq!(view.view_count, 0);
    assert_eq!(view.name, "Bob");
}

#[actix_web::test]
async fn test_full_flow_create_view_check_admin() {
    let data = setup_app_data();
    let app = test::init_service(build_app(data)).await;

    let person = serde_json::json!({
        "name": "Charlie",
        "email": "charlie@test.com",
        "age": 40
    });
    let req = test::TestRequest::post()
        .uri("/persons")
        .set_json(&person)
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 201);
    let created: Person = test::read_body_json(resp).await;
    let person_id = created.id.unwrap();

    let req = test::TestRequest::get()
        .uri(&format!("/persons/{}", person_id))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);

    let req = test::TestRequest::get()
        .uri("/admin/views")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);

    let views: Vec<PersonView> = test::read_body_json(resp).await;
    assert_eq!(views.len(), 1);
    assert_eq!(views[0].view_count, 1);
    assert_eq!(views[0].name, "Charlie");
}

#[actix_web::test]
async fn test_viewing_person_multiple_times_increases_count() {
    let data = setup_app_data();
    let app = test::init_service(build_app(data)).await;

    let person = serde_json::json!({
        "name": "Diana",
        "email": "diana@test.com",
        "age": 35
    });
    let req = test::TestRequest::post()
        .uri("/persons")
        .set_json(&person)
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 201);
    let created: Person = test::read_body_json(resp).await;
    let person_id = created.id.unwrap();

    for _ in 0..5 {
        let req = test::TestRequest::get()
            .uri(&format!("/persons/{}", person_id))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
    }

    let req = test::TestRequest::get()
        .uri(&format!("/admin/views/{}", person_id))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);

    let view: PersonView = test::read_body_json(resp).await;
    assert_eq!(view.view_count, 5);
}

#[actix_web::test]
async fn test_delete_person_removes_from_admin_views() {
    let data = setup_app_data();
    let app = test::init_service(build_app(data)).await;

    let person = serde_json::json!({
        "name": "Eve",
        "email": "eve@test.com",
        "age": 28
    });
    let req = test::TestRequest::post()
        .uri("/persons")
        .set_json(&person)
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 201);
    let created: Person = test::read_body_json(resp).await;
    let person_id = created.id.unwrap();

    let req = test::TestRequest::get()
        .uri(&format!("/persons/{}", person_id))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);

    let req = test::TestRequest::get()
        .uri(&format!("/admin/views/{}", person_id))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
    let view: PersonView = test::read_body_json(resp).await;
    assert_eq!(view.view_count, 1);

    let req = test::TestRequest::delete()
        .uri(&format!("/persons/{}", person_id))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);

    let req = test::TestRequest::get()
        .uri(&format!("/admin/views/{}", person_id))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);

    let req = test::TestRequest::get()
        .uri("/admin/views")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
    let views: Vec<PersonView> = test::read_body_json(resp).await;
    assert_eq!(views.len(), 0);
}

#[actix_web::test]
async fn test_update_person() {
    let data = setup_app_data();
    let app = test::init_service(build_app(data)).await;

    let person = serde_json::json!({
        "name": "Frank",
        "email": "frank@test.com",
        "age": 50
    });
    let req = test::TestRequest::post()
        .uri("/persons")
        .set_json(&person)
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 201);
    let created: Person = test::read_body_json(resp).await;
    let person_id = created.id.unwrap();

    let updated = serde_json::json!({
        "name": "Franklin",
        "email": "franklin@test.com",
        "age": 51
    });
    let req = test::TestRequest::put()
        .uri(&format!("/persons/{}", person_id))
        .set_json(&updated)
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);

    let body: Person = test::read_body_json(resp).await;
    assert_eq!(body.name, "Franklin");
    assert_eq!(body.email, "franklin@test.com");
    assert_eq!(body.age, 51);
}

#[actix_web::test]
async fn test_get_nonexistent_person_returns_404() {
    let data = setup_app_data();
    let app = test::init_service(build_app(data)).await;

    let req = test::TestRequest::get()
        .uri("/persons/9999")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);
}

#[actix_web::test]
async fn test_delete_nonexistent_person_returns_404() {
    let data = setup_app_data();
    let app = test::init_service(build_app(data)).await;

    let req = test::TestRequest::delete()
        .uri("/persons/9999")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);
}

#[actix_web::test]
async fn test_update_nonexistent_person_returns_404() {
    let data = setup_app_data();
    let app = test::init_service(build_app(data)).await;

    let updated = serde_json::json!({
        "name": "Nobody",
        "email": "nobody@test.com",
        "age": 99
    });
    let req = test::TestRequest::put()
        .uri("/persons/9999")
        .set_json(&updated)
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);
}

#[actix_web::test]
async fn test_admin_views_sorted_by_view_count_desc() {
    let data = setup_app_data();
    let app = test::init_service(build_app(data)).await;

    let person_a = serde_json::json!({
        "name": "PersonA",
        "email": "a@test.com",
        "age": 20
    });
    let req = test::TestRequest::post()
        .uri("/persons")
        .set_json(&person_a)
        .to_request();
    let resp = test::call_service(&app, req).await;
    let a: Person = test::read_body_json(resp).await;
    let a_id = a.id.unwrap();

    let person_b = serde_json::json!({
        "name": "PersonB",
        "email": "b@test.com",
        "age": 21
    });
    let req = test::TestRequest::post()
        .uri("/persons")
        .set_json(&person_b)
        .to_request();
    let resp = test::call_service(&app, req).await;
    let b: Person = test::read_body_json(resp).await;
    let b_id = b.id.unwrap();

    for _ in 0..3 {
        let req = test::TestRequest::get()
            .uri(&format!("/persons/{}", b_id))
            .to_request();
        test::call_service(&app, req).await;
    }

    let req = test::TestRequest::get()
        .uri(&format!("/persons/{}", a_id))
        .to_request();
    test::call_service(&app, req).await;

    let req = test::TestRequest::get()
        .uri("/admin/views")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);

    let views: Vec<PersonView> = test::read_body_json(resp).await;
    assert_eq!(views.len(), 2);
    assert_eq!(views[0].name, "PersonB");
    assert_eq!(views[0].view_count, 3);
    assert_eq!(views[1].name, "PersonA");
    assert_eq!(views[1].view_count, 1);
}
