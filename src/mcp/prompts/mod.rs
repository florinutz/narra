//! MCP Prompts for Narra.
//!
//! Prompts provide reusable workflow templates that guide Claude through
//! multi-step processes like consistency checking and dramatic irony analysis.

pub mod character_voice;
pub mod conflict_detection;
pub mod consistency;
pub mod consistency_oracle;
pub mod getting_started;
pub mod irony;
pub mod scene_planning;

pub use character_voice::*;
pub use conflict_detection::*;
pub use consistency::*;
pub use consistency_oracle::*;
pub use getting_started::*;
pub use irony::*;
pub use scene_planning::*;
