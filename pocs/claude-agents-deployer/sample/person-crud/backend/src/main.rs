use actix_cors::Cors;
use actix_web::{web, App, HttpResponse, HttpServer};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Clone)]
struct Person {
    id: Option<i64>,
    name: String,
    email: String,
    age: i32,
}

struct AppState {
    db: Mutex<Connection>,
}

fn init_db(conn: &Connection) {
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
}

async fn get_persons(data: web::Data<AppState>) -> HttpResponse {
    let conn = data.db.lock().unwrap();
    let mut stmt = conn.prepare("SELECT id, name, email, age FROM persons").unwrap();
    let persons: Vec<Person> = stmt
        .query_map([], |row| {
            Ok(Person {
                id: Some(row.get(0)?),
                name: row.get(1)?,
                email: row.get(2)?,
                age: row.get(3)?,
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    HttpResponse::Ok().json(persons)
}

async fn get_person(data: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
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
        Ok(person) => HttpResponse::Ok().json(person),
        Err(_) => HttpResponse::NotFound().json(serde_json::json!({"error": "Not found"})),
    }
}

async fn create_person(data: web::Data<AppState>, person: web::Json<Person>) -> HttpResponse {
    let conn = data.db.lock().unwrap();
    conn.execute(
        "INSERT INTO persons (name, email, age) VALUES (?1, ?2, ?3)",
        (&person.name, &person.email, &person.age),
    )
    .unwrap();
    let id = conn.last_insert_rowid();
    let created = Person {
        id: Some(id),
        name: person.name.clone(),
        email: person.email.clone(),
        age: person.age,
    };
    HttpResponse::Created().json(created)
}

async fn update_person(
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

async fn delete_person(data: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
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

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let conn = Connection::open("persons.db").unwrap();
    init_db(&conn);
    let data = web::Data::new(AppState {
        db: Mutex::new(conn),
    });
    println!("Backend running on http://localhost:8080");
    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();
        App::new()
            .wrap(cors)
            .app_data(data.clone())
            .route("/persons", web::get().to(get_persons))
            .route("/persons/{id}", web::get().to(get_person))
            .route("/persons", web::post().to(create_person))
            .route("/persons/{id}", web::put().to(update_person))
            .route("/persons/{id}", web::delete().to(delete_person))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
