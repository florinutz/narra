//! MCP Resources for Narra.
//!
//! Resources expose static/cached reference data that Claude can access
//! without tool calls. Resources are for reading context, tools are for actions.

mod consistency;
mod entity;
mod operations_guide;
pub mod schema;
mod session;
mod world_overview;

pub use consistency::get_consistency_issues_resource;
pub use entity::get_entity_resource;
pub use operations_guide::get_operations_guide;
pub use schema::{get_import_schema, get_import_template};
pub use session::get_session_context_resource;
pub use world_overview::get_world_overview_resource;
