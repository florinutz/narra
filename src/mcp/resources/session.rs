//! Session context MCP resource.
//!
//! Exposes session state (hot entities, recent activity, pending decisions)
//! as a static resource for Claude to reference without tool calls.

use crate::db::connection::NarraDb;
use std::sync::Arc;

use crate::session::{generate_startup_context, SessionStateManager};

/// Get session context as JSON string for MCP resource.
///
/// Reuses existing generate_startup_context logic from session module.
pub async fn get_session_context_resource(
    session_manager: &Arc<SessionStateManager>,
    db: &Arc<NarraDb>,
) -> Result<String, String> {
    let startup_info = generate_startup_context(session_manager, db)
        .await
        .map_err(|e| format!("Failed to generate session context: {}", e))?;

    serde_json::to_string_pretty(&startup_info)
        .map_err(|e| format!("Failed to serialize session context: {}", e))
}
