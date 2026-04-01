mod models;
mod service;

use actix_cors::Cors;
use actix_web::{web, App, HttpResponse, HttpServer};
use models::{ErrorResponse, HealthResponse, RetirementInput};

async fn calculate(input: web::Json<RetirementInput>) -> HttpResponse {
    if let Err(msg) = service::validate(&input) {
        return HttpResponse::BadRequest().json(ErrorResponse { error: msg });
    }
    let result = service::calculate(&input);
    HttpResponse::Ok().json(result)
}

async fn health() -> HttpResponse {
    HttpResponse::Ok().json(HealthResponse {
        status: "UP".into(),
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Server running on http://localhost:8080");
    HttpServer::new(|| {
        let cors = Cors::default()
            .allowed_origin("http://localhost:5173")
            .allowed_methods(vec!["GET", "POST"])
            .allowed_headers(vec!["Content-Type"])
            .max_age(3600);

        App::new()
            .wrap(cors)
            .route("/api/retirement/calculate", web::post().to(calculate))
            .route("/api/retirement/health", web::get().to(health))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
