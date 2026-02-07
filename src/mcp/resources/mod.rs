//! MCP Resources for Narra.
//!
//! Resources expose static/cached reference data that Claude can access
//! without tool calls. Resources are for reading context, tools are for actions.

mod consistency;
mod entity;
pub mod schema;
mod session;

pub use consistency::get_consistency_issues_resource;
pub use entity::get_entity_resource;
pub use schema::{get_import_schema, get_import_template};
pub use session::get_session_context_resource;
