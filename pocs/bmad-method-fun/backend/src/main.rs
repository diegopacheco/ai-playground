use axum::{
    response::sse::{Event, KeepAlive, Sse},
    routing::get,
    Router,
};
use std::{convert::Infallible, net::SocketAddr, time::Duration};
use tokio_stream::{wrappers::IntervalStream, StreamExt};

async fn config_stream() -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let stream = IntervalStream::new(tokio::time::interval(Duration::from_secs(15)))
        .scan(false, |state, _| {
            *state = !*state;
            let payload = if *state {
                r#"{"background":"matrix","difficulty":"hard","forcedDropIntervalSec":5,"boardExpandIntervalSec":8,"maxLevels":5}"#
            } else {
                r#"{"background":"nebula","difficulty":"normal","forcedDropIntervalSec":40,"boardExpandIntervalSec":30,"maxLevels":10}"#
            };
            Some(Event::default().data(payload))
        })
        .map(Ok);

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/api/config/stream", get(config_stream));
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
