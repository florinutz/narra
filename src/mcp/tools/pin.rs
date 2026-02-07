//! Pin and unpin tools for entity pinning.

use rmcp::handler::server::wrapper::Parameters;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::mcp::NarraServer;

/// Request to pin an entity.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PinEntityRequest {
    /// Entity ID to pin (e.g., "character:alice")
    pub entity_id: String,
}

/// Request to unpin an entity.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UnpinEntityRequest {
    /// Entity ID to unpin
    pub entity_id: String,
}

/// Response for pin/unpin operations.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PinResponse {
    /// Whether the operation succeeded
    pub success: bool,
    /// The entity ID that was pinned/unpinned
    pub entity_id: String,
    /// Current total pinned count
    pub pinned_count: usize,
}

impl NarraServer {
    /// Handler for pin_entity tool - implementation called from server.rs
    pub async fn handle_pin_entity(
        &self,
        Parameters(request): Parameters<PinEntityRequest>,
    ) -> Result<PinResponse, String> {
        self.session_manager.pin_entity(&request.entity_id).await;
        let pinned = self.session_manager.get_pinned().await;
        Ok(PinResponse {
            success: true,
            entity_id: request.entity_id,
            pinned_count: pinned.len(),
        })
    }

    /// Handler for unpin_entity tool - implementation called from server.rs
    pub async fn handle_unpin_entity(
        &self,
        Parameters(request): Parameters<UnpinEntityRequest>,
    ) -> Result<PinResponse, String> {
        self.session_manager.unpin_entity(&request.entity_id).await;
        let pinned = self.session_manager.get_pinned().await;
        Ok(PinResponse {
            success: true,
            entity_id: request.entity_id,
            pinned_count: pinned.len(),
        })
    }
}
