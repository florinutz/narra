use crate::mcp::{EntityResult, MutationResponse, NarraServer};

impl NarraServer {
    pub(crate) async fn handle_create_relationship(
        &self,
        from_character_id: String,
        to_character_id: String,
        rel_type: String,
        subtype: Option<String>,
        label: Option<String>,
    ) -> Result<MutationResponse, String> {
        use crate::models::relationship::{create_relationship, RelationshipCreate};

        let create = RelationshipCreate {
            from_character_id: from_character_id.clone(),
            to_character_id: to_character_id.clone(),
            rel_type: rel_type.clone(),
            subtype: subtype.clone(),
            label: label.clone(),
        };

        let relationship = create_relationship(&self.db, create)
            .await
            .map_err(|e| format!("Failed to create relationship: {}", e))?;

        let entity_id = relationship.id.to_string();

        // Trigger embedding generation for the relates_to edge itself
        self.staleness_manager.spawn_regeneration(
            entity_id.clone(),
            "relates_to".to_string(),
            None,
        );

        // Relationship changes affect character embeddings (composite text includes relationships)
        let from_id = format!("character:{}", from_character_id);
        let to_id = format!("character:{}", to_character_id);
        if let Err(e) = self.staleness_manager.mark_stale(&from_id).await {
            tracing::warn!("Failed to mark embedding stale for {}: {}", from_id, e);
        }
        if let Err(e) = self.staleness_manager.mark_stale(&to_id).await {
            tracing::warn!("Failed to mark embedding stale for {}: {}", to_id, e);
        }
        self.staleness_manager
            .spawn_regeneration(from_id.clone(), "character".to_string(), None);
        self.staleness_manager
            .spawn_regeneration(to_id.clone(), "character".to_string(), None);

        // Mark perspective embeddings stale for perceives edges between these characters
        mark_perspective_stale_for_pair(
            &self.db,
            &self.staleness_manager,
            &from_character_id,
            &to_character_id,
        )
        .await;

        let display_label = label.unwrap_or_else(|| {
            if let Some(ref st) = subtype {
                format!("{}/{}", rel_type, st)
            } else {
                rel_type.clone()
            }
        });

        let result = EntityResult {
            id: entity_id,
            entity_type: "relationship".to_string(),
            name: display_label.clone(),
            content: format!(
                "Created {} relationship: character:{} -> character:{}",
                display_label, from_character_id, to_character_id
            ),
            confidence: Some(1.0),
            last_modified: Some(relationship.created_at.to_string()),
        };

        let hints = vec![
            format!(
                "Relationship '{}' created between character:{} and character:{}",
                display_label, from_character_id, to_character_id
            ),
            "Use graph_traversal to see connected entities".to_string(),
        ];

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_backfill_embeddings(
        &self,
        entity_type: Option<String>,
    ) -> Result<MutationResponse, String> {
        use crate::embedding::BackfillService;

        // Create BackfillService
        let backfill_service =
            BackfillService::new(self.db.clone(), self.embedding_service.clone());

        // Run backfill (either all types or specific type)
        let stats = if let Some(ref etype) = entity_type {
            backfill_service
                .backfill_type(etype)
                .await
                .map_err(|e| format!("Backfill failed: {}", e))?
        } else {
            backfill_service
                .backfill_all()
                .await
                .map_err(|e| format!("Backfill failed: {}", e))?
        };

        // Format stats for response
        let content = if let Some(ref etype) = entity_type {
            format!(
                "Backfill complete for {}: {} total, {} embedded, {} skipped, {} failed",
                etype, stats.total_entities, stats.embedded, stats.skipped, stats.failed
            )
        } else {
            let type_breakdown: Vec<String> = stats
                .entity_type_stats
                .iter()
                .map(|(t, count)| format!("{}: {}", t, count))
                .collect();

            format!(
                "Backfill complete: {} total entities processed, {} embedded, {} skipped, {} failed\n\nBy type:\n{}",
                stats.total_entities,
                stats.embedded,
                stats.skipped,
                stats.failed,
                type_breakdown.join("\n")
            )
        };

        let result = EntityResult {
            id: "backfill_complete".to_string(),
            entity_type: "backfill_operation".to_string(),
            name: "Embedding Backfill".to_string(),
            content,
            confidence: Some(1.0),
            last_modified: Some(chrono::Utc::now().to_rfc3339()),
        };

        let hints = vec![
            format!(
                "Backfill complete. Semantic search now available for {} entities.",
                stats.embedded
            ),
            "Use SemanticSearch to find entities by meaning".to_string(),
            "Use HybridSearch to combine keyword and semantic search".to_string(),
        ];

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_baseline_arc_snapshots(
        &self,
        entity_type: Option<String>,
    ) -> Result<MutationResponse, String> {
        // Validate entity_type if provided
        if let Some(ref et) = entity_type {
            if !matches!(et.as_str(), "character" | "knowledge") {
                return Err(format!(
                    "Invalid entity_type for BaselineArcSnapshots: '{}'. Must be 'character' or 'knowledge'.",
                    et
                ));
            }
        }

        let types_to_process: Vec<String> = match &entity_type {
            Some(et) => vec![et.clone()],
            None => vec!["character".to_string(), "knowledge".to_string()],
        };

        let mut total_created = 0usize;
        let mut total_skipped = 0usize;

        for etype in &types_to_process {
            if !matches!(etype.as_str(), "character" | "knowledge") {
                continue;
            }

            // Fetch entities that have embeddings
            let fetch_query = format!(
                "SELECT id, embedding FROM {} WHERE embedding IS NOT NONE",
                etype
            );

            let mut response = self
                .db
                .query(&fetch_query)
                .await
                .map_err(|e| format!("Failed to fetch {} entities: {}", etype, e))?;

            #[derive(serde::Deserialize)]
            struct EntityWithEmbedding {
                id: surrealdb::RecordId,
                embedding: Vec<f32>,
            }

            let entities: Vec<EntityWithEmbedding> = response
                .take(0)
                .map_err(|e| format!("Failed to parse {} entities: {}", etype, e))?;

            for entity in &entities {
                let entity_id = entity.id.to_string();

                // Check if snapshot already exists
                let mut count_response = self
                    .db
                    .query(
                        "SELECT count() AS cnt FROM arc_snapshot WHERE entity_id = $eid GROUP ALL",
                    )
                    .bind(("eid", entity.id.clone()))
                    .await
                    .map_err(|e| format!("Failed to check snapshots for {}: {}", entity_id, e))?;

                #[derive(serde::Deserialize)]
                struct CountResult {
                    cnt: i64,
                }

                let count: Option<CountResult> = count_response.take(0).unwrap_or(None);
                let existing = count.map(|c| c.cnt).unwrap_or(0);

                if existing > 0 {
                    total_skipped += 1;
                    continue;
                }

                // Create baseline snapshot (no delta_magnitude for first snapshot)
                if let Err(e) = self
                    .db
                    .query(
                        "CREATE arc_snapshot SET entity_id = $eid, entity_type = $etype, embedding = $embedding",
                    )
                    .bind(("eid", entity.id.clone()))
                    .bind(("etype", etype.clone()))
                    .bind(("embedding", entity.embedding.clone()))
                    .await
                {
                    tracing::warn!(
                        "Failed to create baseline snapshot for {}: {}",
                        entity_id,
                        e
                    );
                    continue;
                }

                total_created += 1;
            }
        }

        let content = format!(
            "Baseline arc snapshots: {} created, {} skipped (already had snapshots)",
            total_created, total_skipped
        );

        let result = EntityResult {
            id: "baseline_arc_snapshots".to_string(),
            entity_type: "arc_operation".to_string(),
            name: "Baseline Arc Snapshots".to_string(),
            content,
            confidence: Some(1.0),
            last_modified: Some(chrono::Utc::now().to_rfc3339()),
        };

        let hints = vec![
            format!("Created {} baseline snapshots", total_created),
            "Future embedding regenerations will automatically capture snapshots with delta metrics".to_string(),
            "Use ArcHistory, ArcComparison, ArcDrift, ArcMoment queries to analyze character evolution".to_string(),
        ];

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    // === Consolidated from protect/unprotect standalone tools ===

    pub(crate) async fn handle_protect_entity_mutate(
        &self,
        entity_id: &str,
    ) -> Result<MutationResponse, String> {
        self.impact_service.protect_entity(entity_id).await;
        Ok(MutationResponse {
            entity: EntityResult {
                id: entity_id.to_string(),
                entity_type: "protection".to_string(),
                name: format!("Protected: {}", entity_id),
                content: format!("Entity '{}' is now protected", entity_id),
                confidence: None,
                last_modified: None,
            },
            entities: None,
            impact: None,
            hints: vec!["Entity now triggers CRITICAL severity in impact analysis".to_string()],
        })
    }

    pub(crate) async fn handle_unprotect_entity_mutate(
        &self,
        entity_id: &str,
    ) -> Result<MutationResponse, String> {
        self.impact_service.unprotect_entity(entity_id).await;
        Ok(MutationResponse {
            entity: EntityResult {
                id: entity_id.to_string(),
                entity_type: "protection".to_string(),
                name: format!("Unprotected: {}", entity_id),
                content: format!("Protection removed from entity '{}'", entity_id),
                confidence: None,
                last_modified: None,
            },
            entities: None,
            impact: None,
            hints: vec!["Entity restored to normal severity calculations".to_string()],
        })
    }
}

/// Mark perspective embeddings stale for perceives edges between two characters.
///
/// Called when relationships between characters change, which may affect
/// how they perceive each other.
async fn mark_perspective_stale_for_pair(
    db: &crate::db::connection::NarraDb,
    staleness_manager: &crate::embedding::StalenessManager,
    char_a: &str,
    char_b: &str,
) {
    let ref_a = surrealdb::RecordId::from(("character", char_a));
    let ref_b = surrealdb::RecordId::from(("character", char_b));

    #[derive(serde::Deserialize)]
    struct EdgeId {
        id: surrealdb::RecordId,
    }

    match db
        .query(
            "SELECT id FROM perceives WHERE \
             (in = $ref_a AND out = $ref_b) OR \
             (in = $ref_b AND out = $ref_a)",
        )
        .bind(("ref_a", ref_a))
        .bind(("ref_b", ref_b))
        .await
    {
        Ok(mut resp) => {
            let edges: Vec<EdgeId> = resp.take(0).unwrap_or_default();
            for edge in edges {
                let id_str = edge.id.to_string();
                if let Err(e) = staleness_manager.mark_stale(&id_str).await {
                    tracing::warn!("Failed to mark perspective stale for {}: {}", id_str, e);
                }
                staleness_manager.spawn_regeneration(id_str, "perceives".to_string(), None);
            }
        }
        Err(e) => {
            tracing::warn!(
                "Failed to find perceives edges for pair {}/{}: {}",
                char_a,
                char_b,
                e
            );
        }
    }
}
