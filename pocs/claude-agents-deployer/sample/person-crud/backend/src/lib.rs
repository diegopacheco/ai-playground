use actix_web::{web, HttpResponse};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Clone)]
pub struct Person {
    pub id: Option<i64>,
    pub name: String,
    pub email: String,
    pub age: i32,
}

#[derive(Serialize, Deserialize)]
pub struct PersonView {
    pub id: i64,
    pub name: String,
    pub email: String,
    pub view_count: i64,
}

#[derive(Serialize, Deserialize)]
pub struct PersonWithViews {
    pub id: i64,
    pub name: String,
    pub email: String,
    pub age: i32,
    pub view_count: i64,
}

pub struct AppState {
    pub db: Mutex<Connection>,
}

pub fn init_db(conn: &Connection) {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            age INTEGER NOT NULL
        )",
        [],
    )
    .unwrap();
    conn.execute(
        "CREATE TABLE IF NOT EXISTS post_views (
            person_id INTEGER PRIMARY KEY,
            view_count INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
        )",
        [],
    )
    .unwrap();
    conn.execute_batch("PRAGMA foreign_keys = ON;").unwrap();
}

pub async fn get_persons(data: web::Data<AppState>) -> HttpResponse {
    let conn = data.db.lock().unwrap();
    let mut stmt = conn.prepare(
        "SELECT p.id, p.name, p.email, p.age, COALESCE(pv.view_count, 0) as view_count
         FROM persons p
         LEFT JOIN post_views pv ON p.id = pv.person_id
         ORDER BY p.id"
    ).unwrap();
    let persons: Vec<PersonWithViews> = stmt
        .query_map([], |row| {
            Ok(PersonWithViews {
                id: row.get(0)?,
                name: row.get(1)?,
                email: row.get(2)?,
                age: row.get(3)?,
                view_count: row.get(4)?,
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    HttpResponse::Ok().json(persons)
}

pub async fn get_person(data: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
    let id = path.into_inner();
    let conn = data.db.lock().unwrap();
    let result = conn.query_row(
        "SELECT id, name, email, age FROM persons WHERE id = ?1",
        [id],
        |row| {
            Ok(Person {
                id: Some(row.get(0)?),
                name: row.get(1)?,
                email: row.get(2)?,
                age: row.get(3)?,
            })
        },
    );
    match result {
        Ok(person) => {
            conn.execute(
                "INSERT INTO post_views (person_id, view_count) VALUES (?1, 1)
                 ON CONFLICT(person_id) DO UPDATE SET view_count = view_count + 1",
                [id],
            )
            .unwrap();
            HttpResponse::Ok().json(person)
        }
        Err(_) => HttpResponse::NotFound().json(serde_json::json!({"error": "Not found"})),
    }
}

pub async fn create_person(data: web::Data<AppState>, person: web::Json<Person>) -> HttpResponse {
    let conn = data.db.lock().unwrap();
    conn.execute(
        "INSERT INTO persons (name, email, age) VALUES (?1, ?2, ?3)",
        (&person.name, &person.email, &person.age),
    )
    .unwrap();
    let id = conn.last_insert_rowid();
    conn.execute(
        "INSERT INTO post_views (person_id, view_count) VALUES (?1, 0)",
        [id],
    )
    .unwrap();
    let created = Person {
        id: Some(id),
        name: person.name.clone(),
        email: person.email.clone(),
        age: person.age,
    };
    HttpResponse::Created().json(created)
}

pub async fn update_person(
    data: web::Data<AppState>,
    path: web::Path<i64>,
    person: web::Json<Person>,
) -> HttpResponse {
    let id = path.into_inner();
    let conn = data.db.lock().unwrap();
    let rows = conn
        .execute(
            "UPDATE persons SET name = ?1, email = ?2, age = ?3 WHERE id = ?4",
            (&person.name, &person.email, &person.age, &id),
        )
        .unwrap();
    if rows == 0 {
        HttpResponse::NotFound().json(serde_json::json!({"error": "Not found"}))
    } else {
        let updated = Person {
            id: Some(id),
            name: person.name.clone(),
            email: person.email.clone(),
            age: person.age,
        };
        HttpResponse::Ok().json(updated)
    }
}

pub async fn delete_person(data: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
    let id = path.into_inner();
    let conn = data.db.lock().unwrap();
    let rows = conn
        .execute("DELETE FROM persons WHERE id = ?1", [id])
        .unwrap();
    if rows == 0 {
        HttpResponse::NotFound().json(serde_json::json!({"error": "Not found"}))
    } else {
        HttpResponse::Ok().json(serde_json::json!({"deleted": id}))
    }
}

pub async fn get_admin_views(data: web::Data<AppState>) -> HttpResponse {
    let conn = data.db.lock().unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT p.id, p.name, p.email, COALESCE(pv.view_count, 0)
             FROM persons p
             LEFT JOIN post_views pv ON p.id = pv.person_id
             ORDER BY pv.view_count DESC",
        )
        .unwrap();
    let views: Vec<PersonView> = stmt
        .query_map([], |row| {
            Ok(PersonView {
                id: row.get(0)?,
                name: row.get(1)?,
                email: row.get(2)?,
                view_count: row.get(3)?,
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    HttpResponse::Ok().json(views)
}

pub async fn get_admin_view(data: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
    let id = path.into_inner();
    let conn = data.db.lock().unwrap();
    let result = conn.query_row(
        "SELECT p.id, p.name, p.email, COALESCE(pv.view_count, 0)
         FROM persons p
         LEFT JOIN post_views pv ON p.id = pv.person_id
         WHERE p.id = ?1",
        [id],
        |row| {
            Ok(PersonView {
                id: row.get(0)?,
                name: row.get(1)?,
                email: row.get(2)?,
                view_count: row.get(3)?,
            })
        },
    );
    match result {
        Ok(view) => HttpResponse::Ok().json(view),
        Err(_) => HttpResponse::NotFound().json(serde_json::json!({"error": "Not found"})),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};

    fn setup_test_db() -> web::Data<AppState> {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn);
        web::Data::new(AppState {
            db: Mutex::new(conn),
        })
    }

    #[actix_web::test]
    async fn test_init_db_creates_both_tables() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn);

        let persons_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='persons'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        let post_views_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='post_views'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert!(persons_exists);
        assert!(post_views_exists);
    }

    #[actix_web::test]
    async fn test_create_person_also_creates_post_views_entry() {
        let data = setup_test_db();
        let app = test::init_service(
            App::new()
                .app_data(data.clone())
                .route("/persons", web::post().to(create_person)),
        )
        .await;

        let person = Person {
            id: None,
            name: "Alice".to_string(),
            email: "alice@test.com".to_string(),
            age: 30,
        };
        let req = test::TestRequest::post()
            .uri("/persons")
            .set_json(&person)
            .to_request();
        let resp = test::call_service(&app, req).await;

        assert_eq!(resp.status(), 201);

        let conn = data.db.lock().unwrap();
        let view_count: i64 = conn
            .query_row(
                "SELECT view_count FROM post_views WHERE person_id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(view_count, 0);
    }

    #[actix_web::test]
    async fn test_viewing_person_increments_view_count() {
        let data = setup_test_db();
        let app = test::init_service(
            App::new()
                .app_data(data.clone())
                .route("/persons", web::post().to(create_person))
                .route("/persons/{id}", web::get().to(get_person)),
        )
        .await;

        let person = Person {
            id: None,
            name: "Bob".to_string(),
            email: "bob@test.com".to_string(),
            age: 25,
        };
        let req = test::TestRequest::post()
            .uri("/persons")
            .set_json(&person)
            .to_request();
        test::call_service(&app, req).await;

        let req = test::TestRequest::get()
            .uri("/persons/1")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);

        let req = test::TestRequest::get()
            .uri("/persons/1")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);

        let req = test::TestRequest::get()
            .uri("/persons/1")
            .to_request();
        test::call_service(&app, req).await;

        let conn = data.db.lock().unwrap();
        let view_count: i64 = conn
            .query_row(
                "SELECT view_count FROM post_views WHERE person_id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(view_count, 3);
    }

    #[actix_web::test]
    async fn test_delete_person_cascades_to_post_views() {
        let data = setup_test_db();
        let app = test::init_service(
            App::new()
                .app_data(data.clone())
                .route("/persons", web::post().to(create_person))
                .route("/persons/{id}", web::delete().to(delete_person)),
        )
        .await;

        let person = Person {
            id: None,
            name: "Charlie".to_string(),
            email: "charlie@test.com".to_string(),
            age: 35,
        };
        let req = test::TestRequest::post()
            .uri("/persons")
            .set_json(&person)
            .to_request();
        test::call_service(&app, req).await;

        {
            let conn = data.db.lock().unwrap();
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM post_views WHERE person_id = 1",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(count, 1);
        }

        let req = test::TestRequest::delete()
            .uri("/persons/1")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);

        let conn = data.db.lock().unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM post_views WHERE person_id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }
}
