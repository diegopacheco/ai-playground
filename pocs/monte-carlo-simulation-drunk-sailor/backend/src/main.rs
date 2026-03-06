use actix_cors::Cors;
use actix_web::{web, App, HttpServer, HttpResponse};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct SimulationParams {
    num_simulations: usize,
    pier_length: usize,
}

#[derive(Serialize)]
struct SimulationResult {
    fell_in_water: usize,
    reached_ship: usize,
    total: usize,
    fall_percentage: f64,
    ship_percentage: f64,
    paths: Vec<Vec<i64>>,
}

fn simulate_sailor(pier_length: usize) -> (bool, Vec<i64>) {
    let mut rng = rand::thread_rng();
    let mut position: i64 = pier_length as i64 / 2;
    let mut path = vec![position];

    loop {
        let step: i64 = if rng.gen_bool(0.5) { 1 } else { -1 };
        position += step;
        path.push(position);

        if position <= 0 {
            return (true, path);
        }
        if position >= pier_length as i64 {
            return (false, path);
        }
    }
}

async fn run_simulation(params: web::Json<SimulationParams>) -> HttpResponse {
    let num_simulations = params.num_simulations.min(100_000);
    let pier_length = params.pier_length.max(4).min(200);
    let sample_paths_count = num_simulations.min(50);

    let mut fell_in_water = 0;
    let mut reached_ship = 0;
    let mut paths: Vec<Vec<i64>> = Vec::new();

    for i in 0..num_simulations {
        let (fell, path) = simulate_sailor(pier_length);
        if fell {
            fell_in_water += 1;
        } else {
            reached_ship += 1;
        }
        if i < sample_paths_count {
            paths.push(path);
        }
    }

    let result = SimulationResult {
        fell_in_water,
        reached_ship,
        total: num_simulations,
        fall_percentage: (fell_in_water as f64 / num_simulations as f64) * 100.0,
        ship_percentage: (reached_ship as f64 / num_simulations as f64) * 100.0,
        paths,
    };

    HttpResponse::Ok().json(result)
}

async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({"status": "ok"}))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Backend running on http://localhost:8080");
    HttpServer::new(|| {
        let cors = Cors::permissive();
        App::new()
            .wrap(cors)
            .route("/health", web::get().to(health))
            .route("/simulate", web::post().to(run_simulation))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
