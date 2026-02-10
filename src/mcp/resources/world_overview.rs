//! World overview MCP resource.
//!
//! Exposes a summary of the current world state (entity counts and summaries)
//! as a static resource, equivalent to calling overview(all).

use crate::mcp::NarraServer;

/// Get world overview as JSON string for MCP resource.
///
/// Reuses existing handle_overview logic to produce an overview of all entity types.
pub async fn get_world_overview_resource(server: &NarraServer) -> Result<String, String> {
    let response = server.handle_overview("all", 20).await?;

    serde_json::to_string_pretty(&response)
        .map_err(|e| format!("Failed to serialize world overview: {}", e))
}
