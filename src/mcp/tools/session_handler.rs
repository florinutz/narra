//! Consolidated session tool handler (pin/unpin + get_session_context).

use crate::mcp::{
    HotEntityInfo, NarraServer, PendingDecisionInfo as PendingDecisionInfoType, PinResult,
    SessionContextData, SessionInput, SessionRequest, SessionResponse, WorldOverviewInfo,
};
use crate::session::generate_startup_context;
use rmcp::handler::server::wrapper::Parameters;

impl NarraServer {
    /// Handler for consolidated session tool - dispatches to specific operations.
    pub async fn handle_session(
        &self,
        Parameters(input): Parameters<SessionInput>,
    ) -> Result<SessionResponse, String> {
        // Reconstruct the full request object for deserialization
        let mut full_request = serde_json::Map::new();
        full_request.insert("operation".to_string(), serde_json::json!(input.operation));
        full_request.extend(input.params);

        // Deserialize to SessionRequest
        let request: SessionRequest =
            serde_json::from_value(serde_json::Value::Object(full_request))
                .map_err(|e| format!("Invalid session parameters: {}", e))?;

        match request {
            SessionRequest::GetContext { force_full: _ } => {
                let ctx = self.handle_get_context_session().await?;
                Ok(SessionResponse {
                    operation: "get_context".to_string(),
                    context: Some(ctx),
                    pin_result: None,
                    hints: vec![],
                })
            }
            SessionRequest::PinEntity { entity_id } => {
                let result = self.handle_pin_entity_session(&entity_id).await?;
                Ok(SessionResponse {
                    operation: "pin_entity".to_string(),
                    context: None,
                    pin_result: Some(result),
                    hints: vec![format!("Entity '{}' pinned to working context", entity_id)],
                })
            }
            SessionRequest::UnpinEntity { entity_id } => {
                let result = self.handle_unpin_entity_session(&entity_id).await?;
                Ok(SessionResponse {
                    operation: "unpin_entity".to_string(),
                    context: None,
                    pin_result: Some(result),
                    hints: vec![format!(
                        "Entity '{}' unpinned from working context",
                        entity_id
                    )],
                })
            }
        }
    }

    async fn handle_get_context_session(&self) -> Result<SessionContextData, String> {
        let startup_info = generate_startup_context(&self.session_manager, &self.db)
            .await
            .map_err(|e| format!("Failed to generate session context: {}", e))?;

        Ok(SessionContextData {
            verbosity: startup_info.verbosity.to_string(),
            summary: startup_info.summary,
            last_session_ago: startup_info.last_session_ago,
            hot_entities: startup_info
                .hot_entities
                .into_iter()
                .map(|e| HotEntityInfo {
                    id: e.id,
                    name: e.name,
                    entity_type: e.entity_type,
                    last_accessed: e.last_accessed,
                })
                .collect(),
            pending_decisions: startup_info
                .pending_decisions
                .into_iter()
                .map(|d| PendingDecisionInfoType {
                    id: d.id,
                    description: d.description,
                    age: d.age,
                    affected_count: d.affected_count,
                })
                .collect(),
            world_overview: startup_info.world_overview.map(|o| WorldOverviewInfo {
                character_count: o.character_count,
                location_count: o.location_count,
                event_count: o.event_count,
                scene_count: o.scene_count,
                relationship_count: o.relationship_count,
            }),
        })
    }

    async fn handle_pin_entity_session(&self, entity_id: &str) -> Result<PinResult, String> {
        self.session_manager.pin_entity(entity_id).await;
        let pinned = self.session_manager.get_pinned().await;
        Ok(PinResult {
            success: true,
            entity_id: entity_id.to_string(),
            pinned_count: pinned.len(),
        })
    }

    async fn handle_unpin_entity_session(&self, entity_id: &str) -> Result<PinResult, String> {
        self.session_manager.unpin_entity(entity_id).await;
        let pinned = self.session_manager.get_pinned().await;
        Ok(PinResult {
            success: true,
            entity_id: entity_id.to_string(),
            pinned_count: pinned.len(),
        })
    }
}
