use crate::mcp::{EntityResult, MutationResponse, NarraServer};
use crate::services::{EntityType, PhaseWeights, TemporalService};

fn parse_entity_types(types: Option<Vec<String>>) -> Vec<EntityType> {
    types
        .map(|ts| {
            ts.iter()
                .filter_map(|t| match t.to_lowercase().as_str() {
                    "character" | "char" => Some(EntityType::Character),
                    "location" => Some(EntityType::Location),
                    "event" => Some(EntityType::Event),
                    "scene" => Some(EntityType::Scene),
                    "knowledge" => Some(EntityType::Knowledge),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

impl NarraServer {
    pub(crate) async fn handle_save_phases(
        &self,
        entity_types: Option<Vec<String>>,
        num_phases: Option<usize>,
        content_weight: Option<f32>,
        neighborhood_weight: Option<f32>,
        temporal_weight: Option<f32>,
    ) -> Result<MutationResponse, String> {
        let type_filter = {
            let parsed = parse_entity_types(entity_types);
            if parsed.is_empty() {
                EntityType::embeddable()
            } else {
                parsed
            }
        };

        let weights = if content_weight.is_some()
            || neighborhood_weight.is_some()
            || temporal_weight.is_some()
        {
            Some(PhaseWeights {
                content: content_weight.unwrap_or(0.6),
                neighborhood: neighborhood_weight.unwrap_or(0.25),
                temporal: temporal_weight.unwrap_or(0.15),
            })
        } else {
            None
        };

        let service = TemporalService::new(self.db.clone());

        let result = service
            .detect_phases(type_filter, num_phases, weights)
            .await
            .map_err(|e| format!("Phase detection failed: {}", e))?;

        let phase_count = result.phases.len();

        service
            .save_phases(&result)
            .await
            .map_err(|e| format!("Failed to save phases: {}", e))?;

        Ok(MutationResponse {
            entity: EntityResult {
                id: "phases".to_string(),
                entity_type: "phase_batch".to_string(),
                name: format!("{} phases saved", phase_count),
                content: format!(
                    "Detected and saved {} phases from {} entities",
                    phase_count, result.total_entities
                ),
                confidence: None,
                last_modified: None,
            },
            entities: None,
            hints: vec![
                "Use load_phases to retrieve saved phases instantly".to_string(),
                "Use clear_phases to remove all saved phases".to_string(),
            ],
            impact: None,
        })
    }

    pub(crate) async fn handle_clear_phases(&self) -> Result<MutationResponse, String> {
        let service = TemporalService::new(self.db.clone());

        let count = service
            .delete_all_phases()
            .await
            .map_err(|e| format!("Failed to clear phases: {}", e))?;

        Ok(MutationResponse {
            entity: EntityResult {
                id: "phases".to_string(),
                entity_type: "phase_batch".to_string(),
                name: format!("{} phases cleared", count),
                content: format!("Cleared {} saved phase(s)", count),
                confidence: None,
                last_modified: None,
            },
            entities: None,
            hints: vec!["Use save_phases or detect_phases with save=true to re-detect".to_string()],
            impact: None,
        })
    }
}
