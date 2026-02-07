//! Export tool for world data backup/migration (YAML format).

use chrono::Utc;
use rmcp::handler::server::wrapper::Parameters;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::mcp::NarraServer;
use crate::services::export::ExportService;

/// Request to export world data.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExportRequest {
    /// Output file path (optional, defaults to ./narra-export-{date}.yaml)
    pub output_path: Option<String>,
}

/// Export result with summary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExportResponse {
    /// Path where export was written
    pub output_path: String,
    /// Summary of exported data
    pub summary: ExportSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExportSummary {
    pub character_count: usize,
    pub location_count: usize,
    pub event_count: usize,
    pub scene_count: usize,
    pub relationship_count: usize,
    pub knowledge_count: usize,
    pub note_count: usize,
    pub fact_count: usize,
}

impl NarraServer {
    /// Handler for export_world tool - exports to NarraImport-compatible YAML.
    pub async fn handle_export_world(
        &self,
        Parameters(request): Parameters<ExportRequest>,
    ) -> Result<ExportResponse, String> {
        let export_service = ExportService::new(self.db.clone());
        let import = export_service
            .export_world()
            .await
            .map_err(|e| format!("Export failed: {}", e))?;

        let summary = ExportSummary {
            character_count: import.characters.len(),
            location_count: import.locations.len(),
            event_count: import.events.len(),
            scene_count: import.scenes.len(),
            relationship_count: import.relationships.len(),
            knowledge_count: import.knowledge.len(),
            note_count: import.notes.len(),
            fact_count: import.facts.len(),
        };

        // Serialize to YAML
        let yaml = serde_yaml_ng::to_string(&import)
            .map_err(|e| format!("Failed to serialize export: {}", e))?;

        // Prepend comment header
        let header = format!(
            "# Narra world export\n# Version: {}\n# Exported: {}\n",
            env!("CARGO_PKG_VERSION"),
            Utc::now().to_rfc3339()
        );
        let content = format!("{}{}", header, yaml);

        // Determine output path
        let output_path = request
            .output_path
            .unwrap_or_else(|| format!("./narra-export-{}.yaml", Utc::now().format("%Y-%m-%d")));

        // Write to file
        std::fs::write(&output_path, content)
            .map_err(|e| format!("Failed to write export file: {}", e))?;

        Ok(ExportResponse {
            output_path,
            summary,
        })
    }
}
