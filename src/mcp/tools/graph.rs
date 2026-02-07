//! MCP tool for generating relationship graphs.
//!
//! Generates Mermaid diagrams and writes them to .planning/exports/.

use chrono::Utc;
use rmcp::handler::server::wrapper::Parameters;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::mcp::NarraServer;
use crate::services::graph::{GraphOptions, GraphScope, GraphService, MermaidGraphService};

/// Request to generate a relationship graph.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GraphRequest {
    /// Scope of the graph: "full" for all characters, or "character:ID" for centered view
    pub scope: String,
    /// Depth for character-centered graphs (default: 2)
    #[serde(default)]
    pub depth: Option<usize>,
    /// Include character roles in node labels (default: false)
    #[serde(default)]
    pub include_roles: Option<bool>,
    /// Output filename (optional, auto-generated if not provided)
    #[serde(default)]
    pub filename: Option<String>,
}

/// Response from graph generation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GraphResponse {
    /// Path where the graph was written
    pub file_path: String,
    /// Total characters in output
    pub character_count: usize,
    /// Total relationships in output
    pub relationship_count: usize,
    /// Helpful hints for the user
    pub hints: Vec<String>,
}

impl NarraServer {
    /// Handler for generate_graph tool - implementation called from server.rs
    pub async fn handle_generate_graph(
        &self,
        Parameters(request): Parameters<GraphRequest>,
    ) -> Result<GraphResponse, String> {
        // Parse scope
        let scope = if request.scope.to_lowercase() == "full" {
            GraphScope::FullNetwork
        } else if request.scope.starts_with("character:") {
            let character_id = request
                .scope
                .strip_prefix("character:")
                .unwrap()
                .to_string();
            GraphScope::CharacterCentered {
                character_id,
                depth: request.depth.unwrap_or(2),
            }
        } else {
            return Err(format!(
                "Invalid scope '{}'. Use 'full' or 'character:ID'",
                request.scope
            ));
        };

        // Build options
        let options = GraphOptions {
            include_roles: request.include_roles.unwrap_or(false),
            direction: "TB".to_string(),
        };

        // Create graph service and generate diagram
        let graph_service = MermaidGraphService::new(self.db.clone());
        let mermaid_output = graph_service
            .generate_mermaid(scope.clone(), options)
            .await
            .map_err(|e| format!("Failed to generate graph: {}", e))?;

        // Count characters and relationships from output
        let character_count = mermaid_output.matches("[").count();
        let relationship_count = mermaid_output.matches(" --- ").count();

        // Determine output path
        let filename = request.filename.unwrap_or_else(|| {
            let scope_suffix = match &scope {
                GraphScope::FullNetwork => "full".to_string(),
                GraphScope::CharacterCentered { character_id, .. } => character_id.to_string(),
            };
            format!(
                "graph-{}-{}.md",
                scope_suffix,
                Utc::now().format("%Y%m%d-%H%M%S")
            )
        });

        // Ensure .planning/exports/ directory exists
        let exports_dir = PathBuf::from(".planning/exports");
        std::fs::create_dir_all(&exports_dir)
            .map_err(|e| format!("Failed to create exports directory: {}", e))?;

        let output_path = exports_dir.join(&filename);
        let output_path_str = output_path.to_string_lossy().to_string();

        // Write the file
        std::fs::write(&output_path, &mermaid_output)
            .map_err(|e| format!("Failed to write graph file: {}", e))?;

        // Generate hints
        let hints = match scope {
            GraphScope::FullNetwork => vec![
                format!("Full network graph written to {}", output_path_str),
                "Open in any Mermaid-compatible viewer (GitHub, Obsidian, VSCode)".to_string(),
                "Use scope='character:ID' for a focused view".to_string(),
            ],
            GraphScope::CharacterCentered {
                character_id,
                depth,
            } => vec![
                format!(
                    "Character-centered graph for {} (depth {}) written to {}",
                    character_id, depth, output_path_str
                ),
                "Increase depth parameter to see more distant relationships".to_string(),
            ],
        };

        Ok(GraphResponse {
            file_path: output_path_str,
            character_count,
            relationship_count,
            hints,
        })
    }
}
