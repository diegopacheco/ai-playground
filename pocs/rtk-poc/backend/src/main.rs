use memory_backend::run;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let addr = "127.0.0.1:8080";
    println!("memory backend listening on http://{addr}");
    run(addr).await
}
