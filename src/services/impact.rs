use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use surrealdb::Datetime;

use crate::db::connection::NarraDb;

use crate::repository::{RelationshipRepository, SurrealRelationshipRepository};
use crate::NarraError;

/// Severity level for affected entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Minor ripple effect (distant connection)
    Low,
    /// Moderate impact (needs review)
    Medium,
    /// Significant impact (requires attention)
    High,
    /// Critical - contradiction or protected entity affected
    Critical,
}

/// An entity affected by a change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectedEntity {
    /// Entity ID
    pub id: String,
    /// Entity type
    pub entity_type: String,
    /// Display name
    pub name: String,
    /// Severity of impact
    pub severity: Severity,
    /// Why this entity is affected
    pub reason: String,
    /// Graph distance from changed entity
    pub distance: usize,
    /// Whether this entity is protected
    pub is_protected: bool,
}

/// Impact analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAnalysis {
    /// Changed entity that triggered analysis
    pub changed_entity: String,
    /// Description of the change
    pub change_description: String,
    /// Affected entities grouped by severity
    pub affected_by_severity: HashMap<String, Vec<AffectedEntity>>,
    /// Total count of affected entities
    pub total_affected: usize,
    /// Whether any protected entities are affected
    pub has_protected_impact: bool,
    /// Warnings for protected entities
    pub warnings: Vec<String>,
}

/// A decision record with impact tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    /// Unique decision ID
    pub id: String,
    /// What was decided
    pub description: String,
    /// Reasoning behind the decision
    pub reasoning: String,
    /// When the decision was made
    pub timestamp: Datetime,
    /// Entities affected by this decision
    pub affected_entities: Vec<String>,
    /// Whether implications were fully traced or deferred
    pub implications_traced: bool,
}

/// A deferred implication to be resolved later.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeferredImplication {
    /// Decision that created this implication
    pub decision_id: String,
    /// Entity that may need updating
    pub entity_id: String,
    /// What might need to change
    pub description: String,
    /// When this was deferred
    pub deferred_at: Datetime,
    /// Whether this has been resolved
    pub resolved: bool,
}

/// Impact analysis service trait.
#[async_trait]
pub trait ImpactService: Send + Sync {
    /// Analyze the impact of a change on related entities.
    ///
    /// Returns affected entities grouped by severity level.
    /// Protected entities trigger warnings.
    async fn analyze_impact(
        &self,
        changed_entity: &str,
        change_description: &str,
        max_depth: usize,
    ) -> Result<ImpactAnalysis, NarraError>;

    /// Mark an entity as protected.
    async fn protect_entity(&self, entity_id: &str);

    /// Remove protection from an entity.
    async fn unprotect_entity(&self, entity_id: &str);

    /// Check if an entity is protected.
    async fn is_protected(&self, entity_id: &str) -> bool;

    /// Record a decision with its affected entities.
    async fn record_decision(
        &self,
        description: &str,
        reasoning: &str,
        affected_entities: Vec<String>,
        traced: bool,
    ) -> Decision;

    /// Defer implications for later resolution.
    async fn defer_implications(
        &self,
        decision_id: &str,
        implications: Vec<(String, String)>, // (entity_id, description)
    );

    /// Get all unresolved deferred implications.
    async fn get_deferred_implications(&self) -> Vec<DeferredImplication>;

    /// Mark a deferred implication as resolved.
    async fn resolve_implication(&self, entity_id: &str, decision_id: &str);
}

use std::collections::HashSet;
use tokio::sync::RwLock;
use uuid::Uuid;

/// In-memory implementation of ImpactService.
///
/// Note: In production, decisions and deferred implications would be persisted.
/// This implementation uses in-memory storage for simplicity.
pub struct ImpactAnalyzer {
    db: Arc<NarraDb>,
    relationship_repo: Arc<SurrealRelationshipRepository>,
    /// Protected entity IDs
    protected: RwLock<HashSet<String>>,
    /// Recorded decisions
    decisions: RwLock<Vec<Decision>>,
    /// Deferred implications
    deferred: RwLock<Vec<DeferredImplication>>,
}

impl ImpactAnalyzer {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self {
            db: db.clone(),
            relationship_repo: Arc::new(SurrealRelationshipRepository::new(db)),
            protected: RwLock::new(HashSet::new()),
            decisions: RwLock::new(Vec::new()),
            deferred: RwLock::new(Vec::new()),
        }
    }

    /// Calculate severity based on distance and protection status.
    fn calculate_severity(&self, distance: usize, is_protected: bool) -> Severity {
        calculate_severity(distance, is_protected)
    }

    /// Get entity info for display.
    async fn get_entity_info(&self, entity_id: &str) -> Option<(String, String)> {
        // Parse entity type from ID
        let parts: Vec<&str> = entity_id.split(':').collect();
        if parts.len() != 2 {
            return None;
        }

        let (entity_type, id) = (parts[0], parts[1]);

        // Query for name based on type
        let name_field = match entity_type {
            "character" | "location" => "name",
            "event" | "scene" => "title",
            "knowledge" => "fact",
            _ => return None,
        };

        let query = format!("SELECT {name_field} AS name FROM {entity_type}:{id}");

        let mut response = self.db.query(&query).await.ok()?;

        #[derive(Deserialize)]
        struct NameOnly {
            name: String,
        }

        let result: Option<NameOnly> = response.take(0).ok()?;
        result.map(|r| (entity_type.to_string(), r.name))
    }
}

#[async_trait]
impl ImpactService for ImpactAnalyzer {
    async fn analyze_impact(
        &self,
        changed_entity: &str,
        change_description: &str,
        max_depth: usize,
    ) -> Result<ImpactAnalysis, NarraError> {
        let protected = self.protected.read().await;

        // Check if the changed entity itself is protected
        let changed_is_protected = protected.contains(changed_entity);

        // Get connected entities via graph traversal
        let connected = self
            .relationship_repo
            .get_connected_entities(changed_entity, max_depth)
            .await?;

        // Build affected entities list
        let mut affected_by_severity: HashMap<String, Vec<AffectedEntity>> = HashMap::new();
        let mut warnings: Vec<String> = Vec::new();
        let mut has_protected_impact = changed_is_protected;

        if changed_is_protected {
            warnings.push(format!(
                "WARNING: Entity '{}' itself is protected - changes require explicit review",
                changed_entity
            ));
        }

        // Track distances (BFS order approximates distance)
        for (idx, entity_id) in connected.iter().enumerate() {
            let distance = idx + 1;
            let is_protected = protected.contains(entity_id);

            if is_protected {
                has_protected_impact = true;
                warnings.push(format!(
                    "Protected entity '{}' may be affected by this change",
                    entity_id
                ));
            }

            let severity = self.calculate_severity(distance, is_protected);
            let severity_key = format!("{:?}", severity).to_lowercase();

            // Get entity info for display
            let (entity_type, name) = self
                .get_entity_info(entity_id)
                .await
                .unwrap_or(("unknown".to_string(), entity_id.clone()));

            let reason = if is_protected {
                "Protected entity - requires explicit review".to_string()
            } else {
                format!("Connected via {} relationship(s)", distance)
            };

            let affected = AffectedEntity {
                id: entity_id.clone(),
                entity_type,
                name,
                severity,
                reason,
                distance,
                is_protected,
            };

            affected_by_severity
                .entry(severity_key)
                .or_default()
                .push(affected);
        }

        let total_affected = connected.len();

        Ok(ImpactAnalysis {
            changed_entity: changed_entity.to_string(),
            change_description: change_description.to_string(),
            affected_by_severity,
            total_affected,
            has_protected_impact,
            warnings,
        })
    }

    async fn protect_entity(&self, entity_id: &str) {
        self.protected.write().await.insert(entity_id.to_string());
    }

    async fn unprotect_entity(&self, entity_id: &str) {
        self.protected.write().await.remove(entity_id);
    }

    async fn is_protected(&self, entity_id: &str) -> bool {
        self.protected.read().await.contains(entity_id)
    }

    async fn record_decision(
        &self,
        description: &str,
        reasoning: &str,
        affected_entities: Vec<String>,
        traced: bool,
    ) -> Decision {
        let decision = Decision {
            id: Uuid::new_v4().to_string(),
            description: description.to_string(),
            reasoning: reasoning.to_string(),
            timestamp: Datetime::default(),
            affected_entities,
            implications_traced: traced,
        };

        self.decisions.write().await.push(decision.clone());
        decision
    }

    async fn defer_implications(&self, decision_id: &str, implications: Vec<(String, String)>) {
        let mut deferred = self.deferred.write().await;

        for (entity_id, description) in implications {
            deferred.push(DeferredImplication {
                decision_id: decision_id.to_string(),
                entity_id,
                description,
                deferred_at: Datetime::default(),
                resolved: false,
            });
        }
    }

    async fn get_deferred_implications(&self) -> Vec<DeferredImplication> {
        self.deferred
            .read()
            .await
            .iter()
            .filter(|d| !d.resolved)
            .cloned()
            .collect()
    }

    async fn resolve_implication(&self, entity_id: &str, decision_id: &str) {
        let mut deferred = self.deferred.write().await;

        for imp in deferred.iter_mut() {
            if imp.entity_id == entity_id && imp.decision_id == decision_id {
                imp.resolved = true;
            }
        }
    }
}

/// Standalone severity calculation (extracted for testability).
fn calculate_severity(distance: usize, is_protected: bool) -> Severity {
    if is_protected {
        return Severity::Critical;
    }

    match distance {
        0 => Severity::Critical, // The changed entity itself
        1 => Severity::High,     // Direct connection
        2 => Severity::Medium,   // One hop away
        _ => Severity::Low,      // Distant connection
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_distance_0_is_critical() {
        assert_eq!(calculate_severity(0, false), Severity::Critical);
    }

    #[test]
    fn test_severity_distance_1_is_high() {
        assert_eq!(calculate_severity(1, false), Severity::High);
    }

    #[test]
    fn test_severity_distance_2_is_medium() {
        assert_eq!(calculate_severity(2, false), Severity::Medium);
    }

    #[test]
    fn test_severity_distance_3_plus_is_low() {
        assert_eq!(calculate_severity(3, false), Severity::Low);
        assert_eq!(calculate_severity(10, false), Severity::Low);
        assert_eq!(calculate_severity(100, false), Severity::Low);
    }

    #[test]
    fn test_severity_protected_always_critical() {
        assert_eq!(calculate_severity(0, true), Severity::Critical);
        assert_eq!(calculate_severity(1, true), Severity::Critical);
        assert_eq!(calculate_severity(5, true), Severity::Critical);
    }
}
