use crate::models::{
    Involvement, InvolvementCreate, Perception, PerceptionCreate, Relationship, RelationshipCreate,
    SceneParticipant, SceneParticipantCreate,
};
use crate::NarraError;
use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;
use surrealdb::engine::local::Db;
use surrealdb::Surreal;

/// Repository trait for graph edge operations.
///
/// Covers: Relationship, Perception, SceneParticipant, Involvement edges
#[async_trait]
pub trait RelationshipRepository: Send + Sync {
    // Relationship operations
    async fn create_relationship(
        &self,
        from_id: &str,
        to_id: &str,
        data: RelationshipCreate,
    ) -> Result<Relationship, NarraError>;
    async fn get_character_relationships(
        &self,
        character_id: &str,
    ) -> Result<Vec<Relationship>, NarraError>;

    // Perception operations (asymmetric)
    async fn create_perception(
        &self,
        from_id: &str,
        to_id: &str,
        data: PerceptionCreate,
    ) -> Result<Perception, NarraError>;
    async fn get_perceptions_of(&self, character_id: &str) -> Result<Vec<Perception>, NarraError>;
    async fn get_perceptions_by(&self, character_id: &str) -> Result<Vec<Perception>, NarraError>;

    // Scene participation
    async fn add_scene_participant(
        &self,
        character_id: &str,
        scene_id: &str,
        data: SceneParticipantCreate,
    ) -> Result<SceneParticipant, NarraError>;
    async fn get_scene_participants(
        &self,
        scene_id: &str,
    ) -> Result<Vec<SceneParticipant>, NarraError>;

    // Event involvement
    async fn add_event_involvement(
        &self,
        character_id: &str,
        event_id: &str,
        data: InvolvementCreate,
    ) -> Result<Involvement, NarraError>;
    async fn get_event_participants(&self, event_id: &str) -> Result<Vec<Involvement>, NarraError>;

    // Graph traversal (for context/impact analysis)
    async fn get_connected_entities(
        &self,
        entity_id: &str,
        max_depth: usize,
    ) -> Result<Vec<String>, NarraError>;
}

/// SurrealDB implementation of RelationshipRepository.
pub struct SurrealRelationshipRepository {
    db: Arc<Surreal<Db>>,
}

impl SurrealRelationshipRepository {
    pub fn new(db: Arc<Surreal<Db>>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl RelationshipRepository for SurrealRelationshipRepository {
    async fn create_relationship(
        &self,
        from_id: &str,
        to_id: &str,
        data: RelationshipCreate,
    ) -> Result<Relationship, NarraError> {
        // RelationshipCreate already contains from/to IDs, construct complete struct
        let relationship_data = RelationshipCreate {
            from_character_id: from_id.to_string(),
            to_character_id: to_id.to_string(),
            rel_type: data.rel_type,
            subtype: data.subtype,
            label: data.label,
        };
        crate::models::relationship::create_relationship(&self.db, relationship_data).await
    }

    async fn get_character_relationships(
        &self,
        character_id: &str,
    ) -> Result<Vec<Relationship>, NarraError> {
        crate::models::relationship::get_all_relationships(&self.db, character_id).await
    }

    async fn create_perception(
        &self,
        from_id: &str,
        to_id: &str,
        data: PerceptionCreate,
    ) -> Result<Perception, NarraError> {
        crate::models::perception::create_perception(&self.db, from_id, to_id, data).await
    }

    async fn get_perceptions_of(&self, character_id: &str) -> Result<Vec<Perception>, NarraError> {
        crate::models::perception::get_perceptions_of(&self.db, character_id).await
    }

    async fn get_perceptions_by(&self, character_id: &str) -> Result<Vec<Perception>, NarraError> {
        crate::models::perception::get_perceptions_from(&self.db, character_id).await
    }

    async fn add_scene_participant(
        &self,
        character_id: &str,
        scene_id: &str,
        data: SceneParticipantCreate,
    ) -> Result<SceneParticipant, NarraError> {
        // SceneParticipantCreate already contains character_id and scene_id
        let participant_data = SceneParticipantCreate {
            character_id: character_id.to_string(),
            scene_id: scene_id.to_string(),
            role: data.role,
            notes: data.notes,
        };
        crate::models::scene::add_scene_participant(&self.db, participant_data).await
    }

    async fn get_scene_participants(
        &self,
        scene_id: &str,
    ) -> Result<Vec<SceneParticipant>, NarraError> {
        crate::models::scene::get_scene_participants(&self.db, scene_id).await
    }

    async fn add_event_involvement(
        &self,
        character_id: &str,
        event_id: &str,
        data: InvolvementCreate,
    ) -> Result<Involvement, NarraError> {
        // InvolvementCreate already contains character_id and event_id
        let involvement_data = InvolvementCreate {
            character_id: character_id.to_string(),
            event_id: event_id.to_string(),
            role: data.role,
            impact: data.impact,
        };
        crate::models::scene::add_event_involvement(&self.db, involvement_data).await
    }

    async fn get_event_participants(&self, event_id: &str) -> Result<Vec<Involvement>, NarraError> {
        crate::models::scene::get_event_characters(&self.db, event_id).await
    }

    /// Traverse graph to find all connected entities within max_depth hops.
    ///
    /// Follows multiple edge types: relates_to and perceives (both directions).
    /// Uses one batch query per depth level (one query per frontier entity per edge type),
    /// collecting all results and deduplicating in Rust.
    async fn get_connected_entities(
        &self,
        entity_id: &str,
        max_depth: usize,
    ) -> Result<Vec<String>, NarraError> {
        if max_depth == 0 {
            return Ok(vec![]);
        }

        let mut visited = HashSet::new();
        visited.insert(entity_id.to_string());
        let mut results = Vec::new();
        let mut frontier: Vec<String> = vec![entity_id.to_string()];

        for _depth in 0..max_depth {
            if frontier.is_empty() {
                break;
            }

            // Convert frontier strings to RecordIds for parameterized query
            let refs: Vec<surrealdb::RecordId> = frontier
                .iter()
                .filter_map(|id| {
                    let (table, key) = id.split_once(':')?;
                    Some(surrealdb::RecordId::from((table, key)))
                })
                .collect();

            // Batch all frontier entities into 4 parameterized queries
            let mut response = self
                .db
                .query(
                    "SELECT VALUE out FROM relates_to WHERE in IN $refs;\
                     SELECT VALUE in  FROM relates_to WHERE out IN $refs;\
                     SELECT VALUE out FROM perceives WHERE in IN $refs;\
                     SELECT VALUE in  FROM perceives WHERE out IN $refs",
                )
                .bind(("refs", refs))
                .await?;

            // Collect all neighbors from all 4 query results
            let mut next_frontier = Vec::new();
            for i in 0..4 {
                let neighbors: Vec<surrealdb::sql::Thing> = response.take(i).unwrap_or_default();
                for neighbor in neighbors {
                    let id_str = neighbor.to_string();
                    if visited.insert(id_str.clone()) {
                        results.push(id_str.clone());
                        next_frontier.push(id_str);
                    }
                }
            }

            frontier = next_frontier;
        }

        Ok(results)
    }
}
