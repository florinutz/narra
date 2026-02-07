mod mutate_batch;
mod mutate_entity;
mod mutate_facts;
mod mutate_import;
mod mutate_knowledge;
mod mutate_notes;
mod mutate_ops;

use crate::mcp::{ImpactSummary, MutationRequest, MutationResponse, NarraServer};
use crate::services::{ConsistencySeverity, ImpactAnalysis, ValidationResult};
use rmcp::handler::server::wrapper::Parameters;

impl NarraServer {
    /// Handler for mutate tool - implementation called from server.rs
    pub async fn handle_mutate(
        &self,
        Parameters(request): Parameters<MutationRequest>,
    ) -> Result<MutationResponse, String> {
        match request {
            MutationRequest::CreateCharacter {
                id,
                name,
                role,
                aliases,
                description,
                profile,
            } => {
                self.handle_create_character(id, name, role, aliases, description, profile)
                    .await
            }
            MutationRequest::CreateLocation {
                id,
                name,
                parent_id,
                description,
            } => {
                self.handle_create_location(id, name, description, parent_id)
                    .await
            }
            MutationRequest::CreateEvent {
                id,
                title,
                description,
                sequence,
                date,
                date_precision,
            } => {
                self.handle_create_event(id, title, description, sequence, date, date_precision)
                    .await
            }
            MutationRequest::CreateScene {
                title,
                event_id,
                location_id,
                summary,
            } => {
                self.handle_create_scene(title, summary, event_id, location_id)
                    .await
            }
            MutationRequest::Update { entity_id, fields } => {
                self.handle_update(&entity_id, fields).await
            }
            MutationRequest::RecordKnowledge {
                character_id,
                target_id,
                fact,
                certainty,
                method,
                source_character_id,
                event_id,
            } => {
                self.handle_record_knowledge(
                    character_id,
                    target_id,
                    fact,
                    certainty,
                    method,
                    source_character_id,
                    event_id,
                )
                .await
            }
            MutationRequest::Delete { entity_id, hard } => {
                self.handle_delete(&entity_id, hard.unwrap_or(true)).await
            }
            MutationRequest::CreateNote {
                title,
                body,
                attach_to,
            } => self.handle_create_note(title, body, attach_to).await,
            MutationRequest::AttachNote { note_id, entity_id } => {
                self.handle_attach_note(note_id, entity_id).await
            }
            MutationRequest::DetachNote { note_id, entity_id } => {
                self.handle_detach_note(note_id, entity_id).await
            }
            MutationRequest::CreateFact {
                title,
                description,
                categories,
                enforcement_level,
            } => {
                self.handle_create_fact(title, description, categories, enforcement_level)
                    .await
            }
            MutationRequest::UpdateFact {
                fact_id,
                title,
                description,
                categories,
                enforcement_level,
            } => {
                self.handle_update_fact(fact_id, title, description, categories, enforcement_level)
                    .await
            }
            MutationRequest::DeleteFact { fact_id } => self.handle_delete_fact(fact_id).await,
            MutationRequest::LinkFact { fact_id, entity_id } => {
                self.handle_link_fact(fact_id, entity_id).await
            }
            MutationRequest::UnlinkFact { fact_id, entity_id } => {
                self.handle_unlink_fact(fact_id, entity_id).await
            }
            MutationRequest::CreateRelationship {
                from_character_id,
                to_character_id,
                rel_type,
                subtype,
                label,
            } => {
                self.handle_create_relationship(
                    from_character_id,
                    to_character_id,
                    rel_type,
                    subtype,
                    label,
                )
                .await
            }
            MutationRequest::BatchCreateCharacters { characters } => {
                self.handle_batch_create_characters(characters).await
            }
            MutationRequest::BatchCreateLocations { locations } => {
                self.handle_batch_create_locations(locations).await
            }
            MutationRequest::BatchCreateEvents { events } => {
                self.handle_batch_create_events(events).await
            }
            MutationRequest::BatchCreateRelationships { relationships } => {
                self.handle_batch_create_relationships(relationships).await
            }
            MutationRequest::BackfillEmbeddings { entity_type } => {
                self.handle_backfill_embeddings(entity_type).await
            }
            MutationRequest::BaselineArcSnapshots { entity_type } => {
                self.handle_baseline_arc_snapshots(entity_type).await
            }
            MutationRequest::ProtectEntity { entity_id } => {
                self.handle_protect_entity_mutate(&entity_id).await
            }
            MutationRequest::UnprotectEntity { entity_id } => {
                self.handle_unprotect_entity_mutate(&entity_id).await
            }
            MutationRequest::ImportYaml {
                import,
                on_conflict,
            } => self.handle_import_yaml(import, on_conflict).await,
        }
    }

    /// Check consistency and optionally block on Critical violations.
    /// Returns Err if mutation should be blocked, Ok(warnings) if can proceed.
    pub(crate) fn process_consistency_result(
        &self,
        result: &ValidationResult,
        operation_description: &str,
    ) -> Result<Vec<String>, String> {
        if result.has_blocking_violations {
            // Format CRITICAL violations into error message
            let critical_messages: Vec<String> = result
                .violations_by_severity
                .get(&ConsistencySeverity::Critical)
                .map(|v| v.iter().map(|viol| viol.message.clone()).collect())
                .unwrap_or_default();

            return Err(format!(
                "CRITICAL: {} blocked by universe fact violation(s):\n{}",
                operation_description,
                critical_messages.join("\n")
            ));
        }

        // Collect non-blocking warnings for hints
        Ok(result.warnings())
    }

    pub(crate) fn convert_impact(&self, analysis: &ImpactAnalysis) -> ImpactSummary {
        let severity = if analysis.has_protected_impact {
            "Critical".to_string()
        } else if analysis.total_affected > 10 {
            "High".to_string()
        } else if analysis.total_affected > 3 {
            "Medium".to_string()
        } else {
            "Low".to_string()
        };

        ImpactSummary {
            affected_count: analysis.total_affected,
            severity,
            warnings: analysis.warnings.clone(),
        }
    }

    /// Detect entity type from RecordId format (table:id)
    pub(crate) fn detect_entity_type(&self, entity_id: &str) -> String {
        entity_id.split(':').next().unwrap_or("unknown").to_string()
    }

    /// Extract entity name from ID for display (simplified)
    pub(crate) fn extract_name_from_id(&self, entity_id: &str) -> String {
        entity_id
            .split(':')
            .next_back()
            .unwrap_or(entity_id)
            .to_string()
    }
}
