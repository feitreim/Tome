use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::cache_trie::{CacheTrie, NodeId};
use crate::node_client::NodeConnection;

const W_PREFIX: f64 = 10.0;
const W_LOAD: f64 = 1.0;
const W_QUEUE: f64 = 0.5;

pub struct Scheduler {
    pub nodes: RwLock<HashMap<NodeId, Arc<NodeConnection>>>,
    pub cache_trie: RwLock<CacheTrie>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            cache_trie: RwLock::new(CacheTrie::new()),
        }
    }

    /// Connect to an inference node and register it.
    pub async fn add_node(&self, addr: &str) -> Result<NodeId, String> {
        let conn = NodeConnection::connect(addr)
            .await
            .map_err(|e| format!("failed to connect to {addr}: {e}"))?;

        let node_id = addr.to_string();
        self.nodes
            .write()
            .await
            .insert(node_id.clone(), Arc::new(conn));

        info!(node_id = %node_id, "registered inference node");
        Ok(node_id)
    }

    /// Remove a node from the registry.
    pub async fn remove_node(&self, node_id: &str) {
        self.nodes.write().await.remove(node_id);
        info!(node_id, "removed inference node");
    }

    /// Select the best node for a request using multi-factor scoring.
    pub async fn route_request(&self, tokens: &[u32]) -> Option<(NodeId, Arc<NodeConnection>)> {
        let nodes = self.nodes.read().await;
        if nodes.is_empty() {
            return None;
        }

        let trie = self.cache_trie.read().await;
        let prefix_matches = trie.find_best_match(tokens);

        let mut best_score = f64::NEG_INFINITY;
        let mut best_node: Option<(NodeId, Arc<NodeConnection>)> = None;

        for (node_id, conn) in nodes.iter() {
            let status = conn.status.read().await;
            if !status.healthy {
                continue;
            }

            let prefix_len = prefix_matches
                .iter()
                .find(|(id, _)| id == node_id)
                .map(|(_, len)| *len)
                .unwrap_or(0);

            let load = status.gpu_utilization as f64;
            let queue = status.queue_depth.max(1) as f64;

            let score = W_PREFIX * prefix_len as f64 + W_LOAD * (1.0 - load) + W_QUEUE * (1.0 / queue);

            if score > best_score {
                best_score = score;
                best_node = Some((node_id.clone(), Arc::clone(conn)));
            }
        }

        if best_node.is_none() {
            warn!("no healthy nodes available");
        }
        best_node
    }

    /// Get status info for all registered nodes.
    pub async fn node_statuses(&self) -> Vec<NodeInfo> {
        let nodes = self.nodes.read().await;
        let mut infos = Vec::with_capacity(nodes.len());
        for (id, conn) in nodes.iter() {
            let status = conn.status.read().await;
            infos.push(NodeInfo {
                node_id: id.clone(),
                addr: conn.addr.clone(),
                healthy: status.healthy,
                active_sequences: status.active_sequences,
                gpu_utilization: status.gpu_utilization,
                queue_depth: status.queue_depth,
                cached_tokens: status.cached_tokens,
            });
        }
        infos
    }

    /// Run periodic health checks on all nodes.
    pub async fn health_check_all(&self) {
        let nodes = self.nodes.read().await;
        for (node_id, conn) in nodes.iter() {
            let healthy = conn.health_check().await;
            if !healthy {
                warn!(node_id, "node unhealthy");
            }
        }
    }
}

#[derive(serde::Serialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub addr: String,
    pub healthy: bool,
    pub active_sequences: u32,
    pub gpu_utilization: f32,
    pub queue_depth: u32,
    pub cached_tokens: u64,
}
