use crate::mcp::{EntityResult, MutationResponse, NarraServer};
use crate::models::fact::{
    create_fact, delete_fact, link_fact_to_entity, unlink_fact_from_entity, update_fact,
    EnforcementLevel, FactCategory, FactCreate, FactUpdate,
};

impl NarraServer {
    pub(crate) async fn handle_create_fact(
        &self,
        title: String,
        description: String,
        categories: Option<Vec<String>>,
        enforcement_level: Option<String>,
    ) -> Result<MutationResponse, String> {
        // Parse categories strings to FactCategory enum
        let parsed_categories = categories
            .map(|cats| cats.into_iter().map(|c| Self::parse_category(&c)).collect())
            .unwrap_or_default();

        // Parse enforcement level (default to Warning)
        let parsed_enforcement = enforcement_level
            .map(|e| Self::parse_enforcement_level(&e))
            .unwrap_or(EnforcementLevel::Warning);

        let create = FactCreate {
            title: title.clone(),
            description: description.clone(),
            categories: parsed_categories,
            enforcement_level: parsed_enforcement,
            scope: None, // Scope can be set via update
        };

        let fact = create_fact(&self.db, create)
            .await
            .map_err(|e| format!("Failed to create fact: {}", e))?;

        let entity_id = fact.id.to_string();

        // Trigger embedding generation for the new fact
        if let Err(e) = self.staleness_manager.mark_stale(&entity_id).await {
            tracing::warn!("Failed to mark fact embedding stale: {}", e);
        }
        self.staleness_manager
            .spawn_regeneration(entity_id.clone(), "fact".to_string(), None);

        let result = EntityResult {
            id: entity_id.clone(),
            entity_type: "universe_fact".to_string(),
            name: fact.title.clone(),
            content: format!("Created universe fact: {}", fact.title),
            confidence: Some(1.0),
            last_modified: Some(fact.updated_at.to_string()),
        };

        let hints = vec![
            format!("Universe fact '{}' created successfully", title),
            "Link this fact to entities using graph operations".to_string(),
            format!("Enforcement level: {:?}", parsed_enforcement),
        ];

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_update_fact(
        &self,
        fact_id: String,
        title: Option<String>,
        description: Option<String>,
        categories: Option<Vec<String>>,
        enforcement_level: Option<String>,
    ) -> Result<MutationResponse, String> {
        // Extract fact key from fact_id (handle "universe_fact:xxx" format)
        let fact_key = fact_id.split(':').next_back().unwrap_or(&fact_id);

        // Parse optional categories
        let parsed_categories =
            categories.map(|cats| cats.into_iter().map(|c| Self::parse_category(&c)).collect());

        // Parse optional enforcement level
        let parsed_enforcement = enforcement_level.map(|e| Self::parse_enforcement_level(&e));

        let update = FactUpdate {
            title,
            description,
            categories: parsed_categories,
            enforcement_level: parsed_enforcement,
            scope: None, // Don't update scope in this operation
            updated_at: chrono::Utc::now().into(),
        };

        let fact = update_fact(&self.db, fact_key, update)
            .await
            .map_err(|e| format!("Failed to update fact: {}", e))?
            .ok_or_else(|| format!("Fact not found: {}", fact_id))?;

        let entity_id = fact.id.to_string();

        // Trigger embedding regeneration
        if let Err(e) = self.staleness_manager.mark_stale(&entity_id).await {
            tracing::warn!("Failed to mark fact embedding stale: {}", e);
        }
        self.staleness_manager
            .spawn_regeneration(entity_id.clone(), "fact".to_string(), None);

        let result = EntityResult {
            id: entity_id.clone(),
            entity_type: "universe_fact".to_string(),
            name: fact.title.clone(),
            content: format!("Updated universe fact: {}", fact.title),
            confidence: Some(1.0),
            last_modified: Some(fact.updated_at.to_string()),
        };

        let mut hints = vec![format!(
            "Universe fact '{}' updated successfully",
            fact.title
        )];
        if let Some(ref level) = parsed_enforcement {
            hints.push(format!("Fact enforcement changed to {:?}", level));
        }

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_delete_fact(
        &self,
        fact_id: String,
    ) -> Result<MutationResponse, String> {
        // Extract fact key from fact_id (handle "universe_fact:xxx" format)
        let fact_key = fact_id.split(':').next_back().unwrap_or(&fact_id);

        let fact = delete_fact(&self.db, fact_key)
            .await
            .map_err(|e| format!("Failed to delete fact: {}", e))?
            .ok_or_else(|| format!("Fact not found: {}", fact_id))?;

        let entity_id = fact.id.to_string();

        let result = EntityResult {
            id: entity_id.clone(),
            entity_type: "universe_fact".to_string(),
            name: fact.title.clone(),
            content: format!("Deleted universe fact: {}", fact.title),
            confidence: Some(1.0),
            last_modified: Some(chrono::Utc::now().to_rfc3339()),
        };

        let hints = vec![
            format!("Universe fact '{}' deleted", fact.title),
            "Fact removed, entity links automatically cleaned up".to_string(),
        ];

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_link_fact(
        &self,
        fact_id: String,
        entity_id: String,
    ) -> Result<MutationResponse, String> {
        // Extract fact key from fact_id (handle "universe_fact:xxx" format)
        let fact_key = fact_id.split(':').next_back().unwrap_or(&fact_id);

        // Link with manual link_type (user-initiated)
        let application = link_fact_to_entity(&self.db, fact_key, &entity_id, "manual", None)
            .await
            .map_err(|e| format!("Failed to link fact: {}", e))?;

        // Extract entity type from entity_id for better hint message
        let entity_type = entity_id.split(':').next().unwrap_or("entity");
        let entity_name = entity_id.split(':').next_back().unwrap_or(&entity_id);

        let result = EntityResult {
            id: application.id.to_string(),
            entity_type: "fact_application".to_string(),
            name: format!("Fact {} -> {}", fact_key, entity_id),
            content: format!("Linked fact {} to {}", fact_key, entity_id),
            confidence: Some(1.0),
            last_modified: Some(application.created_at.to_string()),
        };

        let hints = vec![
            format!("Fact now applies to {}: {}", entity_type, entity_name),
            "Use ListFacts with entity_id filter to view facts for this entity".to_string(),
        ];

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_unlink_fact(
        &self,
        fact_id: String,
        entity_id: String,
    ) -> Result<MutationResponse, String> {
        // Extract fact key from fact_id (handle "universe_fact:xxx" format)
        let fact_key = fact_id.split(':').next_back().unwrap_or(&fact_id);

        unlink_fact_from_entity(&self.db, fact_key, &entity_id)
            .await
            .map_err(|e| format!("Failed to unlink fact: {}", e))?;

        let result = EntityResult {
            id: format!("unlink:{}:{}", fact_key, entity_id),
            entity_type: "fact_application".to_string(),
            name: format!("Unlink {} from {}", fact_key, entity_id),
            content: format!("Unlinked fact {} from {}", fact_key, entity_id),
            confidence: Some(1.0),
            last_modified: Some(chrono::Utc::now().to_rfc3339()),
        };

        let hints = vec![format!("Fact no longer applies to {}", entity_id)];

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    /// Parse category string to FactCategory enum
    pub(crate) fn parse_category(s: &str) -> FactCategory {
        match s {
            "physics_magic" => FactCategory::PhysicsMagic,
            "social_cultural" => FactCategory::SocialCultural,
            "technology" => FactCategory::Technology,
            other => FactCategory::Custom(other.to_string()),
        }
    }

    /// Parse enforcement level string to EnforcementLevel enum
    pub(crate) fn parse_enforcement_level(s: &str) -> EnforcementLevel {
        match s.to_lowercase().as_str() {
            "informational" => EnforcementLevel::Informational,
            "warning" => EnforcementLevel::Warning,
            "strict" => EnforcementLevel::Strict,
            _ => EnforcementLevel::Warning, // Default
        }
    }
}
