//! MCP Prompts for Narra.
//!
//! Prompts provide reusable workflow templates that guide Claude through
//! multi-step processes like consistency checking and dramatic irony analysis.

pub mod consistency;
pub mod irony;

pub use consistency::*;
pub use irony::*;
