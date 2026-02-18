use tokio::sync::Mutex;
use tonic::transport::Channel;
use tracing::{info, warn};

use crate::proto::{
    inference_node_client::InferenceNodeClient, GenerateRequest, NodeStatus, PrefillRequest,
    PrefillResponse, StatusRequest, TokenResponse,
};

/// Cached status from the last health check.
#[derive(Clone, Default)]
pub struct CachedStatus {
    pub active_sequences: u32,
    pub gpu_utilization: f32,
    pub queue_depth: u32,
    pub cached_tokens: u64,
    pub healthy: bool,
}

/// Wraps a tonic gRPC client for a single inference node.
pub struct NodeConnection {
    pub addr: String,
    client: Mutex<InferenceNodeClient<Channel>>,
    pub status: tokio::sync::RwLock<CachedStatus>,
}

impl NodeConnection {
    pub async fn connect(addr: &str) -> Result<Self, tonic::transport::Error> {
        let client = InferenceNodeClient::connect(addr.to_string()).await?;
        info!(addr, "connected to inference node");
        Ok(Self {
            addr: addr.to_string(),
            client: Mutex::new(client),
            status: tokio::sync::RwLock::new(CachedStatus {
                healthy: true,
                ..Default::default()
            }),
        })
    }

    pub async fn prefill(&self, req: PrefillRequest) -> Result<PrefillResponse, tonic::Status> {
        let mut client = self.client.lock().await;
        let resp = client.prefill(req).await?;
        Ok(resp.into_inner())
    }

    pub async fn stream_generate(
        &self,
        req: GenerateRequest,
    ) -> Result<tonic::Streaming<TokenResponse>, tonic::Status> {
        let mut client = self.client.lock().await;
        let resp = client.stream_generate(req).await?;
        Ok(resp.into_inner())
    }

    pub async fn get_status(&self) -> Result<NodeStatus, tonic::Status> {
        let mut client = self.client.lock().await;
        let resp = client.get_status(StatusRequest {}).await?;
        let status = resp.into_inner();

        let mut cached = self.status.write().await;
        cached.active_sequences = status.active_sequences;
        cached.gpu_utilization = status.gpu_utilization;
        cached.queue_depth = status.queue_depth;
        cached.cached_tokens = status.cached_tokens;
        cached.healthy = true;

        Ok(status)
    }

    /// Periodic health check. Returns false if the node is unreachable.
    pub async fn health_check(&self) -> bool {
        match self.get_status().await {
            Ok(_) => true,
            Err(e) => {
                warn!(addr = %self.addr, error = %e, "node health check failed");
                let mut cached = self.status.write().await;
                cached.healthy = false;
                false
            }
        }
    }
}
