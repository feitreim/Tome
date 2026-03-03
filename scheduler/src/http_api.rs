use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info};
use uuid::Uuid;

use crate::proto::GenerateRequest;
use crate::scheduler::Scheduler;

const MODEL_ID: &str = "Nanbeige/Nanbeige4.1-3B";

#[derive(Clone)]
pub struct AppState {
    pub scheduler: Arc<Scheduler>,
    pub tokenizer: Arc<Tokenizer>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/grpo", post(grpo))
        .route("/v1/weights", post(update_weights))
        .route("/v1/models", get(list_models))
        .route("/v1/nodes", get(list_nodes).post(register_node))
        .with_state(state)
}

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

fn default_max_tokens() -> u32 {
    128
}
fn default_temperature() -> f32 {
    0.7
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct CompletionChunk {
    id: String,
    choices: Vec<CompletionChoice>,
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let request_id = Uuid::new_v4().to_string();

    let encoding = state
        .tokenizer
        .encode(req.prompt.as_str(), false)
        .expect("tokenizer encode failed");
    let tokens: Vec<u32> = encoding.get_ids().to_vec();

    let (node_id, conn) = match state.scheduler.route_request(&tokens).await {
        Some(pair) => pair,
        None => {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, axum::Error>>(1);
            let _ = tx
                .send(Ok(Event::default().data(
                    serde_json::to_string(&serde_json::json!({
                        "error": "no healthy inference nodes available"
                    }))
                    .unwrap(),
                )))
                .await;
            return Sse::new(ReceiverStream::new(rx));
        }
    };

    info!(request_id, node_id, "routing completion request");

    let grpc_req = GenerateRequest {
        request_id: request_id.clone(),
        tokens,
        temperature: req.temperature,
        max_tokens: req.max_tokens,
    };

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, axum::Error>>(32);
    let rid = request_id.clone();
    let tokenizer = Arc::clone(&state.tokenizer);

    tokio::spawn(async move {
        let mut token_ids: Vec<u32> = Vec::new();
        let mut decoded_len: usize = 0;

        match conn.stream_generate(grpc_req).await {
            Ok(mut stream) => {
                while let Some(result) = stream.message().await.transpose() {
                    match result {
                        Ok(token_resp) => {
                            let finish_reason = if token_resp.is_finished {
                                Some("stop".to_string())
                            } else {
                                None
                            };

                            token_ids.push(token_resp.token_id);
                            let text = match tokenizer.decode(&token_ids, true) {
                                Ok(decoded) => {
                                    let new_text = decoded[decoded_len..].to_string();
                                    decoded_len = decoded.len();
                                    new_text
                                }
                                Err(_) => String::new(),
                            };

                            let chunk = CompletionChunk {
                                id: rid.clone(),
                                choices: vec![CompletionChoice {
                                    text,
                                    finish_reason,
                                }],
                            };

                            let data = serde_json::to_string(&chunk).unwrap();
                            if tx.send(Ok(Event::default().data(data))).await.is_err() {
                                break;
                            }

                            if token_resp.is_finished {
                                break;
                            }
                        }
                        Err(e) => {
                            error!(request_id = %rid, error = %e, "stream error");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                error!(request_id = %rid, error = %e, "failed to start generation");
                let _ = tx
                    .send(Ok(Event::default().data(
                        serde_json::to_string(&serde_json::json!({ "error": e.to_string() }))
                            .unwrap(),
                    )))
                    .await;
            }
        }
    });

    Sse::new(ReceiverStream::new(rx))
}

// ---- OpenAI-compatible chat completions (SSE) ----

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    model: &'static str,
    choices: Vec<ChatChunkChoice>,
}

#[derive(Serialize)]
struct ChatChunkChoice {
    index: u32,
    delta: ChatDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!(
                    "<|im_start|>assistant\n{}<|im_end|>\n",
                    msg.content
                ));
            }
            _ => {}
        }
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Sse<ReceiverStream<Result<Event, axum::Error>>>, (StatusCode, Json<serde_json::Value>)>
{
    let request_id = Uuid::new_v4().to_string();
    let prompt = format_chat_prompt(&req.messages);

    let encoding = state
        .tokenizer
        .encode(prompt.as_str(), false)
        .expect("tokenizer encode failed");
    let tokens: Vec<u32> = encoding.get_ids().to_vec();

    let (node_id, conn) = match state.scheduler.route_request(&tokens).await {
        Some(pair) => pair,
        None => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": {
                        "message": "no healthy inference nodes available",
                        "type": "server_error",
                        "code": "service_unavailable"
                    }
                })),
            ));
        }
    };

    info!(request_id, node_id, "routing chat completion request");

    let grpc_req = GenerateRequest {
        request_id: request_id.clone(),
        tokens,
        temperature: req.temperature,
        max_tokens: req.max_tokens,
    };

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, axum::Error>>(32);
    let rid = request_id.clone();
    let tokenizer = Arc::clone(&state.tokenizer);

    // Send initial chunk with role
    let init_chunk = ChatCompletionChunk {
        id: rid.clone(),
        object: "chat.completion.chunk",
        model: MODEL_ID,
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant"),
                content: None,
            },
            finish_reason: None,
        }],
    };
    let _ = tx
        .send(Ok(Event::default()
            .data(serde_json::to_string(&init_chunk).unwrap())))
        .await;

    tokio::spawn(async move {
        let mut token_ids: Vec<u32> = Vec::new();
        let mut decoded_len: usize = 0;

        match conn.stream_generate(grpc_req).await {
            Ok(mut stream) => {
                while let Some(result) = stream.message().await.transpose() {
                    match result {
                        Ok(token_resp) => {
                            let finish_reason = if token_resp.is_finished {
                                Some("stop")
                            } else {
                                None
                            };

                            token_ids.push(token_resp.token_id);
                            let text = match tokenizer.decode(&token_ids, true) {
                                Ok(decoded) => {
                                    let new_text = decoded[decoded_len..].to_string();
                                    decoded_len = decoded.len();
                                    new_text
                                }
                                Err(_) => String::new(),
                            };

                            let chunk = ChatCompletionChunk {
                                id: rid.clone(),
                                object: "chat.completion.chunk",
                                model: MODEL_ID,
                                choices: vec![ChatChunkChoice {
                                    index: 0,
                                    delta: ChatDelta {
                                        role: None,
                                        content: Some(text),
                                    },
                                    finish_reason,
                                }],
                            };

                            let data = serde_json::to_string(&chunk).unwrap();
                            if tx.send(Ok(Event::default().data(data))).await.is_err() {
                                break;
                            }

                            if token_resp.is_finished {
                                let _ = tx
                                    .send(Ok(Event::default().data("[DONE]")))
                                    .await;
                                break;
                            }
                        }
                        Err(e) => {
                            error!(request_id = %rid, error = %e, "stream error");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                error!(request_id = %rid, error = %e, "failed to start generation");
                let _ = tx
                    .send(Ok(Event::default().data(
                        serde_json::to_string(&serde_json::json!({ "error": e.to_string() }))
                            .unwrap(),
                    )))
                    .await;
            }
        }
    });

    Ok(Sse::new(ReceiverStream::new(rx)))
}

// ---- Models endpoint ----

async fn list_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "owned_by": "tome"
        }]
    }))
}

async fn list_nodes(State(state): State<AppState>) -> Json<serde_json::Value> {
    let nodes = state.scheduler.node_statuses().await;
    Json(serde_json::json!({ "nodes": nodes }))
}

#[derive(Deserialize)]
struct RegisterNodeRequest {
    addr: String,
}

async fn register_node(
    State(state): State<AppState>,
    Json(req): Json<RegisterNodeRequest>,
) -> impl IntoResponse {
    match state.scheduler.add_node(&req.addr).await {
        Ok(node_id) => Json(serde_json::json!({ "node_id": node_id, "status": "registered" })),
        Err(e) => Json(serde_json::json!({ "error": e })),
    }
}

// ---- GRPO API ----

#[derive(Deserialize)]
struct GrpoRequestJson {
    batch_id: String,
    prompts: Vec<GrpoPromptJson>,
    group_size: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_grpo_max_tokens")]
    max_tokens: u32,
    max_concurrent: Option<u32>,
    judge: JudgeConfigJson,
}

#[derive(Deserialize)]
struct GrpoPromptJson {
    prompt_id: String,
    prompt: String,
}

#[derive(Deserialize)]
struct JudgeConfigJson {
    rubric: String,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_grpo_max_tokens")]
    max_tokens: u32,
}

fn default_grpo_max_tokens() -> u32 {
    512
}

async fn grpo(
    State(state): State<AppState>,
    Json(req): Json<GrpoRequestJson>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let request_id = Uuid::new_v4().to_string();

    // Encode rubric and prompts
    let rubric_tokens = state
        .tokenizer
        .encode(req.judge.rubric.as_str(), false)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))))?
        .get_ids()
        .to_vec();

    let mut proto_prompts = Vec::new();
    for p in req.prompts {
        let tokens = state
            .tokenizer
            .encode(p.prompt.as_str(), false)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))))?
            .get_ids()
            .to_vec();
        proto_prompts.push(crate::proto::Prompt {
            prompt_id: p.prompt_id,
            tokens,
        });
    }

    // Route based on rubric (shared prefix)
    let (node_id, conn) = match state.scheduler.route_request(&rubric_tokens).await {
        Some(pair) => pair,
        None => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": "no healthy inference nodes available" })),
            ));
        }
    };

    info!(request_id, node_id, "routing GRPO request");

    let grpc_req = crate::proto::GrpoRequest {
        batch_id: req.batch_id,
        prompts: proto_prompts,
        group_size: req.group_size,
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        max_concurrent: req.max_concurrent.unwrap_or(4),
        judge: Some(crate::proto::JudgeConfig {
            rubric_tokens,
            temperature: req.judge.temperature,
            max_tokens: req.judge.max_tokens,
        }),
    };

    match conn.grpo(grpc_req).await {
        Ok(resp) => Ok(Json(serde_json::to_value(resp).unwrap())),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() })))),
    }
}

// ---- Weight Update API ----

#[derive(Deserialize)]
struct WeightUpdateRequestJson {
    updates: Vec<LayerUpdateJson>,
}

#[derive(Deserialize)]
struct LayerUpdateJson {
    layer_idx: u32,
    param_name: String,
    lora_a: String, // base64 encoded
    lora_b: String, // base64 encoded
    shape_a: Vec<u32>,
    shape_b: Vec<u32>,
}

async fn update_weights(
    State(state): State<AppState>,
    Json(req): Json<WeightUpdateRequestJson>,
) -> Json<serde_json::Value> {
    use base64::{engine::general_purpose, Engine as _};

    let mut proto_updates = Vec::new();
    for u in req.updates {
        let lora_a = general_purpose::STANDARD.decode(u.lora_a).unwrap_or_default();
        let lora_b = general_purpose::STANDARD.decode(u.lora_b).unwrap_or_default();
        proto_updates.push(crate::proto::LayerUpdate {
            layer_idx: u.layer_idx,
            param_name: u.param_name,
            lora_a,
            lora_b,
            shape_a: u.shape_a,
            shape_b: u.shape_b,
        });
    }

    let grpc_req = crate::proto::WeightUpdateRequest {
        updates: proto_updates,
    };

    let results = state.scheduler.broadcast_weight_update(grpc_req).await;
    
    let mut response_results = serde_json::Map::new();
    for (node_id, res) in results {
        match res {
            Ok(resp) => {
                response_results.insert(node_id, serde_json::json!({ "success": resp.success, "version": resp.policy_version }));
            }
            Err(e) => {
                response_results.insert(node_id, serde_json::json!({ "error": e }));
            }
        }
    }

    Json(serde_json::Value::Object(response_results))
}
