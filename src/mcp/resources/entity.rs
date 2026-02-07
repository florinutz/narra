//! Entity view MCP resource.
//!
//! Exposes full entity details (attributes, relationships, knowledge)
//! via URI template narra://entity/{type}:{id}.

use std::sync::Arc;

use crate::services::context::ContextService;

/// Get entity full view as JSON string for MCP resource.
///
/// Delegates to ContextService.get_entity_full_detail for business logic.
///
/// # Arguments
/// * `entity_id` - Full entity identifier (e.g., "character:alice")
/// * `context_service` - Context service for entity lookup
///
/// # Returns
/// JSON string with full entity content, or error if not found.
pub async fn get_entity_resource(
    entity_id: &str,
    context_service: &Arc<dyn ContextService + Send + Sync>,
) -> Result<String, String> {
    let entity = context_service
        .get_entity_full_detail(entity_id)
        .await
        .map_err(|e| format!("Failed to fetch entity: {}", e))?
        .ok_or_else(|| format!("Entity not found: {}", entity_id))?;

    serde_json::to_string_pretty(&entity).map_err(|e| format!("Failed to serialize entity: {}", e))
}
