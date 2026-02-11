use crate::mcp::types::KnowledgeSpec;
use crate::mcp::{EntityResult, MutationResponse, NarraServer};
use crate::models::{CertaintyLevel, KnowledgeStateCreate, LearningMethod};

impl NarraServer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn handle_record_knowledge(
        &self,
        character_id: String,
        target_id: String,
        fact: String,
        certainty: String,
        method: Option<String>,
        source_character_id: Option<String>,
        event_id: Option<String>,
    ) -> Result<MutationResponse, String> {
        use crate::models::knowledge::{create_knowledge, create_knowledge_state, KnowledgeCreate};
        use surrealdb::RecordId;

        // Build creation data for consistency check
        let creation_data = serde_json::json!({
            "character_id": character_id,
            "target_id": target_id,
            "fact": fact,
            "certainty": certainty,
        });

        // Check consistency but NEVER block - knowledge represents character beliefs
        let consistency_result = self
            .consistency_service
            .check_entity_creation("knowledge", &creation_data)
            .await
            .map_err(|e| format!("Consistency check failed: {}", e))?;

        // For knowledge, we don't block even on Critical - collect all as info hints
        let mut hints = Vec::new();
        if consistency_result.total_violations > 0 {
            hints.push("Note: This knowledge contradicts universe fact(s) - may be intentional dramatic irony".to_string());
            // Add violation details as info, not warnings
            for (severity, violations) in &consistency_result.violations_by_severity {
                for v in violations {
                    hints.push(format!(
                        "  - {} ({:?}): {}",
                        v.fact_title, severity, v.message
                    ));
                }
            }
        }

        // Step 1: Create knowledge entity with the fact
        // The knowledge entity stores: character (the knower) + fact (the information)
        // The target_id from the API is incorporated into the fact text
        let char_record_id: RecordId = character_id
            .parse()
            .map_err(|e| format!("Invalid character_id: {}", e))?;

        let knowledge_entity = create_knowledge(
            &self.db,
            KnowledgeCreate {
                character: char_record_id.clone(),
                fact: fact.clone(),
            },
        )
        .await
        .map_err(|e| format!("Failed to create knowledge entity: {}", e))?;

        // Step 2: Parse certainty level and learning method
        let certainty_level = match certainty.to_lowercase().as_str() {
            "knows" | "certain" => CertaintyLevel::Knows,
            "suspects" => CertaintyLevel::Suspects,
            "believes_wrongly" | "wrong" => CertaintyLevel::BelievesWrongly,
            "uncertain" => CertaintyLevel::Uncertain,
            "assumes" => CertaintyLevel::Assumes,
            "denies" => CertaintyLevel::Denies,
            "forgotten" => CertaintyLevel::Forgotten,
            _ => CertaintyLevel::Uncertain,
        };

        let learning_method = method.map(|m| match m.to_lowercase().as_str() {
            "told" | "direct" => LearningMethod::Told,
            "overheard" => LearningMethod::Overheard,
            "witnessed" => LearningMethod::Witnessed,
            "discovered" => LearningMethod::Discovered,
            "deduced" => LearningMethod::Deduced,
            "read" => LearningMethod::Read,
            "remembered" => LearningMethod::Remembered,
            "initial" => LearningMethod::Initial,
            _ => LearningMethod::Discovered,
        });

        // Clone event_id before it's moved into KnowledgeStateCreate (for arc snapshot)
        let arc_event_id = event_id.clone();

        // Step 3: Create knows edge from character to knowledge entity
        let create = KnowledgeStateCreate {
            certainty: certainty_level,
            learning_method: learning_method.unwrap_or(LearningMethod::Discovered),
            source_character: source_character_id,
            event: event_id,
            premises: None,
            truth_value: if certainty_level == CertaintyLevel::BelievesWrongly {
                Some(fact.clone())
            } else {
                None
            },
        };

        // Extract character key (function expects just key, not table:key)
        let char_key = character_id.split(':').nth(1).unwrap_or(&character_id);
        let knowledge_target = format!("knowledge:{}", knowledge_entity.id.key());

        let knowledge_state = create_knowledge_state(&self.db, char_key, &knowledge_target, create)
            .await
            .map_err(|e| format!("Failed to create knowledge state: {}", e))?;

        // Trigger async embedding generation for the knowledge entity
        let knowledge_entity_id = knowledge_entity.id.to_string();
        self.staleness_manager.spawn_regeneration(
            knowledge_entity_id,
            "knowledge".to_string(),
            arc_event_id,
        );

        let entity_id = knowledge_state.id.to_string();

        let result = EntityResult {
            id: entity_id,
            entity_type: "knowledge".to_string(),
            name: fact.clone(),
            content: format!(
                "{} now {:?} about {}: {}",
                character_id, certainty_level, target_id, fact
            ),
            confidence: Some(1.0),
            last_modified: Some(knowledge_state.learned_at.to_string()),
        };

        // Prepend primary knowledge confirmation, add query suggestion
        hints.insert(
            0,
            format!("Knowledge recorded: {} about {}", character_id, target_id),
        );
        hints.push("Query temporal knowledge to see knowledge history".to_string());

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_batch_record_knowledge(
        &self,
        knowledge: Vec<KnowledgeSpec>,
    ) -> Result<MutationResponse, String> {
        let count = knowledge.len();
        let mut entities = Vec::with_capacity(count);
        let mut errors: Vec<String> = Vec::new();

        for spec in knowledge {
            match self
                .handle_record_knowledge(
                    spec.character_id.clone(),
                    spec.target_id.clone(),
                    spec.fact.clone(),
                    spec.certainty,
                    spec.method,
                    spec.source_character_id,
                    spec.event_id,
                )
                .await
            {
                Ok(response) => {
                    entities.push(response.entity);
                }
                Err(e) => {
                    errors.push(format!(
                        "Failed to record knowledge for {} about {}: {}",
                        spec.character_id, spec.target_id, e
                    ));
                }
            }
        }

        if entities.is_empty() && !errors.is_empty() {
            return Err(format!(
                "All {} knowledge entries failed: {}",
                count,
                errors.join("; ")
            ));
        }

        let mut hints = vec![format!(
            "Recorded {}/{} knowledge entries",
            entities.len(),
            count
        )];
        if !errors.is_empty() {
            hints.push(format!("Errors: {}", errors.join("; ")));
        }
        hints.push("Query temporal knowledge to see knowledge history".to_string());

        let summary = EntityResult {
            id: String::new(),
            entity_type: "batch".to_string(),
            name: format!("Batch: {} knowledge entries", entities.len()),
            content: format!("Recorded {} knowledge entries", entities.len()),
            confidence: Some(1.0),
            last_modified: None,
        };

        Ok(MutationResponse {
            entity: summary,
            entities: Some(entities),
            impact: None,
            hints,
        })
    }
}
