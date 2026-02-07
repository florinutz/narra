use crate::mcp::types::{ConflictMode, NarraImport};
use crate::mcp::{EntityResult, MutationResponse, NarraServer};
use crate::services::import::ImportService;

impl NarraServer {
    pub(crate) async fn handle_import_yaml(
        &self,
        import: NarraImport,
        on_conflict: ConflictMode,
    ) -> Result<MutationResponse, String> {
        let import_service = ImportService::new(self.db.clone(), self.staleness_manager.clone());

        let result = import_service
            .execute_import(import, on_conflict)
            .await
            .map_err(|e| format!("Import failed: {}", e))?;

        // Build summary text
        let summary_lines: Vec<String> = result
            .by_type
            .iter()
            .filter(|t| t.created > 0 || t.skipped > 0 || t.updated > 0 || !t.errors.is_empty())
            .map(|t| {
                let mut parts = vec![format!("{}: {} created", t.entity_type, t.created)];
                if t.skipped > 0 {
                    parts.push(format!("{} skipped", t.skipped));
                }
                if t.updated > 0 {
                    parts.push(format!("{} updated", t.updated));
                }
                if !t.errors.is_empty() {
                    parts.push(format!("{} errors", t.errors.len()));
                }
                parts.join(", ")
            })
            .collect();

        let summary_text = if summary_lines.is_empty() {
            "No entities imported".to_string()
        } else {
            summary_lines.join("\n")
        };

        let total = result.total_created + result.total_skipped + result.total_updated;
        let entity = EntityResult {
            id: String::new(),
            entity_type: "import".to_string(),
            name: format!("Import: {} entities", total),
            content: summary_text,
            confidence: Some(1.0),
            last_modified: None,
        };

        let mut hints = vec![format!(
            "Created: {}, Skipped: {}, Updated: {}, Errors: {}",
            result.total_created, result.total_skipped, result.total_updated, result.total_errors,
        )];

        // Add error details
        for type_result in &result.by_type {
            for err in &type_result.errors {
                hints.push(format!("[{}] {}", type_result.entity_type, err));
            }
        }

        if result.total_created > 0 {
            hints.push(
                "Run mutate(BackfillEmbeddings) to generate embeddings for imported entities"
                    .to_string(),
            );
        }

        Ok(MutationResponse {
            entity,
            entities: None,
            impact: None,
            hints,
        })
    }
}
