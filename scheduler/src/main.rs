mod cache_trie;
mod http_api;
mod node_client;
mod proto;
mod scheduler;

use std::sync::Arc;

use tokenizers::Tokenizer;
use tracing::info;

use crate::http_api::AppState;
use crate::scheduler::Scheduler;

const MODEL_ID: &str = "Nanbeige/Nanbeige4.1-3B";

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "scheduler=info".into()),
        )
        .init();

    info!("Loading tokenizer from {MODEL_ID}...");
    let tokenizer = Tokenizer::from_pretrained(MODEL_ID, None)
        .expect("failed to load tokenizer");

    let scheduler = Arc::new(Scheduler::new());

    let state = AppState {
        scheduler: Arc::clone(&scheduler),
        tokenizer: Arc::new(tokenizer),
    };

    let app = http_api::router(state);

    // Spawn periodic health checks
    let health_scheduler = Arc::clone(&scheduler);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            interval.tick().await;
            health_scheduler.health_check_all().await;
        }
    });

    let port = std::env::var("SCHEDULER_PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{port}");
    info!("scheduler listening on {addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
