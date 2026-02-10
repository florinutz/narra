use crate::db::connection::NarraDb;
use crate::models::{
    CertaintyLevel, Knowledge, KnowledgeConflict, KnowledgeCreate, KnowledgeState,
    KnowledgeStateCreate, KnowledgeTransmission,
};
use crate::NarraError;
use async_trait::async_trait;
use std::sync::Arc;

/// Repository trait for knowledge and knowledge state operations.
///
/// Covers: Knowledge facts, KnowledgeState edges, temporal queries, provenance
#[async_trait]
pub trait KnowledgeRepository: Send + Sync {
    // Knowledge (fact) operations
    async fn create_knowledge(&self, data: KnowledgeCreate) -> Result<Knowledge, NarraError>;
    async fn get_knowledge(&self, id: &str) -> Result<Option<Knowledge>, NarraError>;
    async fn get_character_knowledge(
        &self,
        character_id: &str,
    ) -> Result<Vec<Knowledge>, NarraError>;
    async fn search_knowledge(&self, query: &str) -> Result<Vec<Knowledge>, NarraError>;

    // Knowledge state (edge) operations
    async fn create_knowledge_state(
        &self,
        character_id: &str,
        target: &str,
        data: KnowledgeStateCreate,
    ) -> Result<KnowledgeState, NarraError>;
    async fn get_character_knowledge_states(
        &self,
        character_id: &str,
    ) -> Result<Vec<KnowledgeState>, NarraError>;
    async fn get_fact_knowers(&self, target: &str) -> Result<Vec<KnowledgeState>, NarraError>;
    async fn update_certainty(
        &self,
        character_id: &str,
        target: &str,
        new_certainty: CertaintyLevel,
        event_id: &str,
    ) -> Result<KnowledgeState, NarraError>;

    // Temporal queries
    async fn get_knowledge_at_event(
        &self,
        character_id: &str,
        event_id: &str,
    ) -> Result<Vec<KnowledgeState>, NarraError>;
    async fn get_knowledge_history(
        &self,
        character_id: &str,
        target: &str,
    ) -> Result<Vec<KnowledgeState>, NarraError>;
    async fn get_current_knowledge(
        &self,
        character_id: &str,
        target: &str,
    ) -> Result<Option<KnowledgeState>, NarraError>;

    // Provenance queries
    async fn get_transmission_chain(
        &self,
        target: &str,
    ) -> Result<Vec<KnowledgeTransmission>, NarraError>;
    async fn find_conflicts(&self) -> Result<Vec<KnowledgeConflict>, NarraError>;
    async fn get_possible_sources(
        &self,
        knowledge_state_id: &str,
    ) -> Result<Vec<String>, NarraError>;
}

/// SurrealDB implementation of KnowledgeRepository.
pub struct SurrealKnowledgeRepository {
    db: Arc<NarraDb>,
}

impl SurrealKnowledgeRepository {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl KnowledgeRepository for SurrealKnowledgeRepository {
    async fn create_knowledge(&self, data: KnowledgeCreate) -> Result<Knowledge, NarraError> {
        crate::models::knowledge::create_knowledge(&self.db, data).await
    }

    async fn get_knowledge(&self, id: &str) -> Result<Option<Knowledge>, NarraError> {
        crate::models::knowledge::get_knowledge(&self.db, id).await
    }

    async fn get_character_knowledge(
        &self,
        character_id: &str,
    ) -> Result<Vec<Knowledge>, NarraError> {
        crate::models::knowledge::get_character_knowledge(&self.db, character_id).await
    }

    async fn search_knowledge(&self, query: &str) -> Result<Vec<Knowledge>, NarraError> {
        crate::models::knowledge::search_knowledge_by_fact(&self.db, query).await
    }

    async fn create_knowledge_state(
        &self,
        character_id: &str,
        target: &str,
        data: KnowledgeStateCreate,
    ) -> Result<KnowledgeState, NarraError> {
        crate::models::knowledge::create_knowledge_state(&self.db, character_id, target, data).await
    }

    async fn get_character_knowledge_states(
        &self,
        character_id: &str,
    ) -> Result<Vec<KnowledgeState>, NarraError> {
        crate::models::knowledge::get_character_knowledge_states(&self.db, character_id).await
    }

    async fn get_fact_knowers(&self, target: &str) -> Result<Vec<KnowledgeState>, NarraError> {
        crate::models::knowledge::get_fact_knowers(&self.db, target).await
    }

    async fn update_certainty(
        &self,
        character_id: &str,
        target: &str,
        new_certainty: CertaintyLevel,
        event_id: &str,
    ) -> Result<KnowledgeState, NarraError> {
        crate::models::knowledge::update_knowledge_certainty(
            &self.db,
            character_id,
            target,
            new_certainty,
            event_id,
        )
        .await
    }

    async fn get_knowledge_at_event(
        &self,
        character_id: &str,
        event_id: &str,
    ) -> Result<Vec<KnowledgeState>, NarraError> {
        crate::models::knowledge::get_knowledge_at_event(&self.db, character_id, event_id).await
    }

    async fn get_knowledge_history(
        &self,
        character_id: &str,
        target: &str,
    ) -> Result<Vec<KnowledgeState>, NarraError> {
        crate::models::knowledge::get_knowledge_history(&self.db, character_id, target).await
    }

    async fn get_current_knowledge(
        &self,
        character_id: &str,
        target: &str,
    ) -> Result<Option<KnowledgeState>, NarraError> {
        crate::models::knowledge::get_current_knowledge(&self.db, character_id, target).await
    }

    async fn get_transmission_chain(
        &self,
        target: &str,
    ) -> Result<Vec<KnowledgeTransmission>, NarraError> {
        crate::models::knowledge::get_transmission_chain(&self.db, target).await
    }

    async fn find_conflicts(&self) -> Result<Vec<KnowledgeConflict>, NarraError> {
        crate::models::knowledge::find_knowledge_conflicts(&self.db).await
    }

    async fn get_possible_sources(
        &self,
        knowledge_state_id: &str,
    ) -> Result<Vec<String>, NarraError> {
        crate::models::knowledge::get_possible_sources(&self.db, knowledge_state_id).await
    }
}
