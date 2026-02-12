//! MCP-specific progress reporter implementation.
//!
//! Wraps `Peer<RoleServer>` and `ProgressToken` to send MCP progress
//! notifications. Created from tool `Meta` when the client requests progress.

use async_trait::async_trait;
use rmcp::model::{ProgressNotificationParam, ProgressToken};
use rmcp::{Peer, RoleServer};

use crate::services::progress::ProgressReporter;

/// MCP progress reporter that sends progress notifications to the client.
pub struct McpProgressReporter {
    client: Peer<RoleServer>,
    token: ProgressToken,
}

impl McpProgressReporter {
    /// Create from the progress token and MCP client peer.
    pub fn new(client: Peer<RoleServer>, token: ProgressToken) -> Self {
        Self { client, token }
    }
}

#[async_trait]
impl ProgressReporter for McpProgressReporter {
    async fn report(&self, current: f64, total: f64, message: Option<String>) {
        let _ = self
            .client
            .notify_progress(ProgressNotificationParam {
                progress_token: self.token.clone(),
                progress: current,
                total: Some(total),
                message,
            })
            .await;
    }
}

/// Create a progress reporter from MCP Meta + Peer, falling back to noop.
///
/// Usage in `#[tool]` methods:
/// ```ignore
/// let progress = make_mcp_progress(&meta, &client);
/// self.handle_dossier(id, progress).await
/// ```
pub fn make_mcp_progress(
    meta: &rmcp::model::Meta,
    client: &Peer<RoleServer>,
) -> std::sync::Arc<dyn ProgressReporter> {
    match meta.get_progress_token() {
        Some(token) => std::sync::Arc::new(McpProgressReporter::new(client.clone(), token.clone())),
        None => crate::services::progress::noop_progress(),
    }
}
