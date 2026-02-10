//! Consistency checking service for universe facts.
//!
//! This service detects violations of universe facts during entity mutations,
//! providing severity-based warnings that can block operations or allow them
//! to proceed with user awareness.

use crate::db::connection::NarraDb;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::{timeout, Duration};

use crate::models::event::get_event;
use crate::models::fact::{
    get_entity_facts, list_facts, EnforcementLevel, PovScope, TemporalScope, UniverseFact,
};
use crate::models::knowledge::get_character_knowledge_states;
use crate::models::perception::{get_perception, get_perceptions_from};
use crate::models::scene::{get_character_scenes, get_scene};
use crate::NarraError;

// ============================================================================
// Severity Types
// ============================================================================

/// Severity level for consistency violations.
///
/// Ordered from most to least severe: Critical > Warning > Info.
/// This ordering enables sorting violations by importance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ConsistencySeverity {
    /// Informational only - low confidence or informational enforcement
    Info,
    /// Shows warning, allows mutation - warning enforcement + high confidence
    Warning,
    /// Blocks mutation - strict enforcement + high confidence
    Critical,
}

// ============================================================================
// Violation Types
// ============================================================================

/// A single consistency violation detected during validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    /// ID of the fact that was violated
    pub fact_id: String,
    /// Title of the violated fact for display
    pub fact_title: String,
    /// Severity level of this violation
    pub severity: ConsistencySeverity,
    /// Human-readable message describing the violation
    pub message: String,
    /// Confidence score of the violation detection (0.0-1.0)
    pub confidence: f32,
    /// Whether this was auto-detected as intentional (e.g., dramatic irony)
    pub auto_detected_as_intentional: bool,
}

// ============================================================================
// Validation Result Types
// ============================================================================

/// Aggregated validation result grouping violations by severity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the validation passed (no Critical violations)
    pub is_valid: bool,
    /// Violations grouped by severity level
    pub violations_by_severity: HashMap<ConsistencySeverity, Vec<Violation>>,
    /// Total count of all violations
    pub total_violations: usize,
    /// Whether any Critical violations are present
    pub has_blocking_violations: bool,
}

impl ValidationResult {
    /// Create a new empty validation result (valid, no violations).
    pub fn new() -> Self {
        Self {
            is_valid: true,
            violations_by_severity: HashMap::new(),
            total_violations: 0,
            has_blocking_violations: false,
        }
    }

    /// Add a violation to the result.
    ///
    /// Updates is_valid and has_blocking_violations based on severity.
    pub fn add_violation(&mut self, violation: Violation) {
        let severity = violation.severity;

        // Update blocking status
        if severity == ConsistencySeverity::Critical {
            self.is_valid = false;
            self.has_blocking_violations = true;
        }

        // Add to violations_by_severity
        self.violations_by_severity
            .entry(severity)
            .or_default()
            .push(violation);

        // Update total count
        self.total_violations += 1;
    }

    /// Get formatted warning messages for MCP response.
    ///
    /// Returns Warning and Info level violations as formatted strings.
    /// Critical violations are typically handled separately (blocking).
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Collect Warning-level violations
        if let Some(warning_violations) = self
            .violations_by_severity
            .get(&ConsistencySeverity::Warning)
        {
            for v in warning_violations {
                warnings.push(format!("WARNING: {}", v.message));
            }
        }

        // Collect Info-level violations
        if let Some(info_violations) = self.violations_by_severity.get(&ConsistencySeverity::Info) {
            for v in info_violations {
                warnings.push(format!("INFO: {}", v.message));
            }
        }

        warnings
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Severity Mapping and Helper Functions
// ============================================================================

/// Generate a suggested fix for a violation.
pub fn generate_suggested_fix(violation: &Violation) -> Option<String> {
    if violation.message.contains("before learning") {
        Some("Move the learning event to before this scene, or remove character from scene".into())
    } else if violation.message.contains("Circular parent")
        || violation.message.contains("Circular child")
    {
        Some(
            "Remove one parent/child relationship - characters cannot be mutual parents/children"
                .into(),
        )
    } else if violation.message.contains("asymmetry") || violation.message.contains("Asymmetric") {
        Some("Review if asymmetry is intentional dramatic tension or an error".into())
    } else if violation.message.contains("may violate fact") {
        Some("Review entity against the universe fact, update entity or adjust fact scope".into())
    } else {
        None
    }
}

/// Map enforcement level and confidence to severity.
///
/// Logic:
/// - Intentional violations (dramatic irony, etc.) -> Info
/// - Strict enforcement + high confidence (>0.5) -> Critical
/// - Warning enforcement + high confidence (>0.5) -> Warning
/// - All other cases -> Info
pub fn map_enforcement_to_severity(
    enforcement: EnforcementLevel,
    confidence: f32,
    is_intentional: bool,
) -> ConsistencySeverity {
    if is_intentional {
        return ConsistencySeverity::Info;
    }
    match enforcement {
        EnforcementLevel::Strict if confidence > 0.5 => ConsistencySeverity::Critical,
        EnforcementLevel::Warning if confidence > 0.5 => ConsistencySeverity::Warning,
        _ => ConsistencySeverity::Info,
    }
}

// ============================================================================
// Service Trait
// ============================================================================

/// Consistency checking service trait.
///
/// Detects violations of universe facts during entity mutations and creations.
#[async_trait]
pub trait ConsistencyService: Send + Sync {
    /// Check entity mutation for fact violations.
    ///
    /// Returns all violations grouped by severity.
    ///
    /// # Arguments
    ///
    /// * `entity_id` - Full entity identifier (e.g., "character:alice")
    /// * `mutation_fields` - JSON object with fields being mutated
    ///
    /// # Returns
    ///
    /// ValidationResult with all detected violations.
    async fn check_entity_mutation(
        &self,
        entity_id: &str,
        mutation_fields: &serde_json::Value,
    ) -> Result<ValidationResult, NarraError>;

    /// Check entity creation for fact violations.
    ///
    /// # Arguments
    ///
    /// * `entity_type` - Type of entity being created ("character", "location", etc.)
    /// * `creation_data` - JSON object with entity creation data
    ///
    /// # Returns
    ///
    /// ValidationResult with all detected violations.
    async fn check_entity_creation(
        &self,
        entity_type: &str,
        creation_data: &serde_json::Value,
    ) -> Result<ValidationResult, NarraError>;

    /// Check timeline violations for a character.
    ///
    /// Detects when a character knows something at a scene that occurs
    /// before the event where they learned it.
    async fn check_timeline_violations(
        &self,
        character_id: &str,
    ) -> Result<Vec<Violation>, NarraError>;

    /// Check relationship violations for a character.
    ///
    /// Detects impossible states and asymmetric perceptions.
    async fn check_relationship_violations(
        &self,
        character_id: &str,
    ) -> Result<Vec<Violation>, NarraError>;

    /// Investigate contradictions across connected entities.
    ///
    /// Traverses the entity graph up to max_depth hops, checking each entity
    /// for fact, timeline, and relationship violations.
    async fn investigate_contradictions(
        &self,
        entity_id: &str,
        max_depth: usize,
    ) -> Result<(Vec<Violation>, usize), NarraError>;
}

// ============================================================================
// ConsistencyChecker Implementation
// ============================================================================

/// Consistency checker that validates entities against universe facts.
pub struct ConsistencyChecker {
    db: Arc<NarraDb>,
}

impl ConsistencyChecker {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self { db }
    }

    /// Get directly connected entity IDs via graph edges.
    ///
    /// Queries SurrealDB for entities connected via:
    /// - perceives (character -> character)
    /// - knows (character -> knowledge)
    /// - involved_in (character -> scene)
    /// - has (location -> location hierarchy)
    /// - applies_to (fact -> entity)
    async fn get_connected_entities(&self, entity_id: &str) -> Result<Vec<String>, NarraError> {
        let entity_type = entity_id.split(':').next().unwrap_or("");
        let entity_key = entity_id.split(':').nth(1).unwrap_or(entity_id);

        let mut connected: Vec<String> = Vec::new();

        // Query outgoing edges based on entity type
        // SurrealDB edge traversal: SELECT ->edge_type->target FROM entity_id
        match entity_type {
            "character" => {
                // Outgoing perceives edges
                let query = format!(
                    "SELECT VALUE out FROM perceives WHERE in = character:{}",
                    entity_key
                );
                let mut result = self
                    .db
                    .query(&query)
                    .await
                    .map_err(|e| NarraError::Database(e.to_string()))?;
                let targets: Vec<surrealdb::sql::Thing> = result
                    .take(0)
                    .map_err(|e| NarraError::Database(e.to_string()))?;
                for t in targets {
                    connected.push(t.to_string());
                }

                // Scenes character is involved in
                let query = format!(
                    "SELECT VALUE out FROM involved_in WHERE in = character:{}",
                    entity_key
                );
                let mut result = self
                    .db
                    .query(&query)
                    .await
                    .map_err(|e| NarraError::Database(e.to_string()))?;
                let targets: Vec<surrealdb::sql::Thing> = result
                    .take(0)
                    .map_err(|e| NarraError::Database(e.to_string()))?;
                for t in targets {
                    connected.push(t.to_string());
                }
            }
            "scene" => {
                // Characters in scene (reverse involved_in)
                let query = format!(
                    "SELECT VALUE in FROM involved_in WHERE out = scene:{}",
                    entity_key
                );
                let mut result = self
                    .db
                    .query(&query)
                    .await
                    .map_err(|e| NarraError::Database(e.to_string()))?;
                let targets: Vec<surrealdb::sql::Thing> = result
                    .take(0)
                    .map_err(|e| NarraError::Database(e.to_string()))?;
                for t in targets {
                    connected.push(t.to_string());
                }
            }
            "fact" => {
                // Entities this fact applies to
                let query = format!(
                    "SELECT VALUE out FROM applies_to WHERE in = fact:{}",
                    entity_key
                );
                let mut result = self
                    .db
                    .query(&query)
                    .await
                    .map_err(|e| NarraError::Database(e.to_string()))?;
                let targets: Vec<surrealdb::sql::Thing> = result
                    .take(0)
                    .map_err(|e| NarraError::Database(e.to_string()))?;
                for t in targets {
                    connected.push(t.to_string());
                }
            }
            _ => {}
        }

        Ok(connected)
    }

    /// Get facts that apply to this entity (linked via applies_to).
    /// Falls back to all Strict facts if no specific links exist.
    async fn get_applicable_facts(&self, entity_id: &str) -> Result<Vec<UniverseFact>, NarraError> {
        // First try facts directly linked to this entity
        let linked_facts = get_entity_facts(&self.db, entity_id).await?;

        if !linked_facts.is_empty() {
            return Ok(linked_facts);
        }

        // If no linked facts, return Strict facts (they apply globally by default)
        let all_facts = list_facts(&self.db).await?;
        Ok(all_facts
            .into_iter()
            .filter(|f| f.enforcement_level == EnforcementLevel::Strict)
            .collect())
    }

    /// Evaluate a single fact against entity data.
    /// Returns Some(Violation) if fact is violated, None otherwise.
    ///
    /// `entity_id` and `current_event_sequence` are used for scope filtering:
    /// if the fact's scope excludes this entity or time context, the fact is skipped.
    async fn evaluate_fact(
        &self,
        fact: &UniverseFact,
        entity_id: &str,
        entity_data: &serde_json::Value,
        is_intentional: bool,
        current_event_sequence: Option<i64>,
    ) -> Option<Violation> {
        // Check scope — if fact doesn't apply to this entity/time, skip it
        if !self
            .is_fact_in_scope(fact, entity_id, entity_data, current_event_sequence)
            .await
        {
            return None;
        }

        // Simple text matching: check if entity data contradicts fact
        // This is a heuristic approach - looks for negation patterns
        let data_str = entity_data.to_string().to_lowercase();
        let fact_title_lower = fact.title.to_lowercase();
        let fact_desc_lower = fact.description.to_lowercase();

        // Check for contradiction patterns
        let has_potential_violation =
            self.detect_potential_violation(&data_str, &fact_title_lower, &fact_desc_lower);

        if has_potential_violation {
            let confidence = self.calculate_confidence(&data_str, &fact_title_lower);
            let severity =
                map_enforcement_to_severity(fact.enforcement_level, confidence, is_intentional);

            Some(Violation {
                fact_id: fact.id.to_string(),
                fact_title: fact.title.clone(),
                severity,
                message: format!(
                    "Entity may violate fact: {}. {}",
                    fact.title, fact.description
                ),
                confidence,
                auto_detected_as_intentional: is_intentional,
            })
        } else {
            None
        }
    }

    /// Check whether a fact's scope includes the given entity and time context.
    /// Returns `true` if the fact should be evaluated against this entity.
    async fn is_fact_in_scope(
        &self,
        fact: &UniverseFact,
        entity_id: &str,
        entity_data: &serde_json::Value,
        current_event_sequence: Option<i64>,
    ) -> bool {
        let scope = match &fact.scope {
            Some(s) => s,
            None => return true, // No scope = applies globally
        };

        // Check POV scope
        if let Some(ref pov) = scope.pov {
            if !Self::check_pov_scope(pov, entity_id, entity_data) {
                return false;
            }
        }

        // Check temporal scope
        if let Some(ref temporal) = scope.temporal {
            if !self
                .check_temporal_scope(temporal, current_event_sequence)
                .await
            {
                return false;
            }
        }

        true
    }

    /// Check whether the entity matches the fact's POV scope.
    fn check_pov_scope(pov: &PovScope, entity_id: &str, entity_data: &serde_json::Value) -> bool {
        match pov {
            PovScope::Character(char_id) => {
                // Only applies if entity_id matches the specified character
                entity_id == char_id.as_str()
                    || entity_id.ends_with(&format!(":{}", char_id))
                    || char_id.ends_with(&format!(":{}", entity_id.split(':').nth(1).unwrap_or("")))
            }
            PovScope::Group(group_name) => {
                // Check if entity has this group in roles or categories
                let roles = entity_data
                    .get("roles")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .any(|r| r.as_str().map(|s| s == group_name).unwrap_or(false))
                    })
                    .unwrap_or(false);
                roles
            }
            PovScope::ExceptCharacters(excluded_ids) => {
                // Applies to all EXCEPT listed characters
                !excluded_ids.iter().any(|ex| {
                    entity_id == ex.as_str()
                        || entity_id.ends_with(&format!(":{}", ex.split(':').nth(1).unwrap_or(ex)))
                })
            }
        }
    }

    /// Check whether the current time context falls within the fact's temporal scope.
    async fn check_temporal_scope(
        &self,
        temporal: &TemporalScope,
        current_event_sequence: Option<i64>,
    ) -> bool {
        let current_seq = match current_event_sequence {
            Some(seq) => seq,
            None => return true, // No event context = can't exclude by time
        };

        // Check valid_from_event
        if let Some(ref from_event_id) = temporal.valid_from_event {
            if let Some(from_seq) = self.resolve_event_sequence(from_event_id).await {
                if current_seq < from_seq {
                    return false; // Before the fact becomes valid
                }
            }
        }

        // Check valid_until_event
        if let Some(ref until_event_id) = temporal.valid_until_event {
            if let Some(until_seq) = self.resolve_event_sequence(until_event_id).await {
                if current_seq > until_seq {
                    return false; // After the fact expires
                }
            }
        }

        true
    }

    /// Resolve an event ID to its sequence number.
    async fn resolve_event_sequence(&self, event_id: &str) -> Option<i64> {
        let event_key = event_id.split(':').nth(1).unwrap_or(event_id);
        get_event(&self.db, event_key)
            .await
            .ok()
            .flatten()
            .map(|e| e.sequence)
    }

    /// Get the latest event sequence number in the world.
    async fn get_latest_event_sequence(&self) -> Option<i64> {
        let mut result = self
            .db
            .query("SELECT VALUE sequence FROM event ORDER BY sequence DESC LIMIT 1")
            .await
            .ok()?;
        let seqs: Vec<i64> = result.take(0).ok()?;
        seqs.into_iter().next()
    }

    /// Detect if entity data potentially violates a fact.
    /// Uses keyword and negation pattern matching.
    fn detect_potential_violation(
        &self,
        entity_data: &str,
        fact_title: &str,
        fact_description: &str,
    ) -> bool {
        // Extract key concepts from fact (simple word extraction)
        let fact_keywords: Vec<&str> = fact_title
            .split_whitespace()
            .chain(fact_description.split_whitespace())
            .filter(|w| w.len() > 3) // Skip small words
            .collect();

        // Look for negation patterns near fact keywords
        let negation_words = ["no ", "not ", "cannot ", "never ", "without ", "lacks "];

        for keyword in &fact_keywords {
            // Check if keyword appears in entity data with negation
            if let Some(pos) = entity_data.find(*keyword) {
                // Check for negation within 20 chars before the keyword
                let start = pos.saturating_sub(20);
                let context = &entity_data[start..pos];

                for neg in &negation_words {
                    if context.contains(neg) {
                        return true;
                    }
                }
            }
        }

        // Also check for explicit contradictions mentioned in fact
        // e.g., fact says "no magic" but entity has "magic ability"
        if fact_title.starts_with("no ") || fact_description.contains("prohibited") {
            let forbidden_term = fact_title.trim_start_matches("no ").trim();
            if entity_data.contains(forbidden_term) {
                return true;
            }
        }

        false
    }

    /// Calculate confidence score based on match quality.
    fn calculate_confidence(&self, entity_data: &str, fact_title: &str) -> f32 {
        // Simple heuristic: more keyword matches = higher confidence
        let keywords: Vec<&str> = fact_title
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        if keywords.is_empty() {
            return 0.5;
        }

        let matches = keywords.iter().filter(|k| entity_data.contains(*k)).count();

        let base_confidence = matches as f32 / keywords.len() as f32;

        // Scale to 0.3-0.9 range (never fully certain with heuristics)
        0.3 + (base_confidence * 0.6)
    }

    /// Check if a violation might be intentional (dramatic irony).
    /// Returns true if the entity is a character's belief/knowledge.
    async fn is_intentional_contradiction(
        &self,
        entity_id: &str,
        _entity_data: &serde_json::Value,
    ) -> bool {
        // If this is a knowledge entity or the mutation is about beliefs,
        // it's likely intentional (dramatic irony, unreliable narrator)
        let entity_type = entity_id.split(':').next().unwrap_or("");

        // Knowledge entities are character beliefs - contradictions are intentional
        entity_type == "knowledge"
    }

    /// Check if a knowledge target relates to a Strict enforcement fact.
    async fn is_related_to_strict_fact(&self, target: &str) -> Result<bool, NarraError> {
        // Get facts linked to this target
        let facts = get_entity_facts(&self.db, target).await?;
        Ok(facts
            .iter()
            .any(|f| f.enforcement_level == EnforcementLevel::Strict))
    }

    /// Check for timeline violations in character knowledge.
    ///
    /// Detects when a character knows something at a scene that occurs
    /// before the event where they learned it.
    ///
    /// Per CONTEXT.md:
    /// - Strict ordering enforced (can't know X before learning X)
    /// - CRITICAL if violation relates to Strict-enforcement fact
    /// - WARNING otherwise
    pub async fn check_timeline_violations(
        &self,
        character_id: &str,
    ) -> Result<Vec<Violation>, NarraError> {
        let mut violations = Vec::new();

        // Get all knowledge states for this character
        let knowledge_states = get_character_knowledge_states(&self.db, character_id).await?;

        // Get all scene participations for this character
        let participations = get_character_scenes(&self.db, character_id).await?;

        // For each knowledge state with a learning event
        for knowledge_state in &knowledge_states {
            if let Some(learning_event_id) = &knowledge_state.event {
                // Get the learning event to find its sequence number
                let learning_event_key = learning_event_id.key().to_string();

                if let Some(learning_event) = get_event(&self.db, &learning_event_key).await? {
                    let learning_sequence = learning_event.sequence;

                    // Check each scene where character participates
                    for participation in &participations {
                        // Get the scene to access its event
                        let scene_key = participation.scene.key().to_string();
                        if let Some(scene) = get_scene(&self.db, &scene_key).await? {
                            // Get the scene's event to check sequence
                            let scene_event_key = scene.event.key().to_string();

                            if let Some(scene_event) = get_event(&self.db, &scene_event_key).await?
                            {
                                // Violation if scene happens before learning event
                                if scene_event.sequence < learning_sequence {
                                    // Extract target ID for fact checking
                                    let target_key = knowledge_state.target.key().to_string();
                                    let target_full = knowledge_state.target.to_string();

                                    // Check if this relates to a Strict fact
                                    let is_strict =
                                        self.is_related_to_strict_fact(&target_full).await?;
                                    let severity = if is_strict {
                                        ConsistencySeverity::Critical
                                    } else {
                                        ConsistencySeverity::Warning
                                    };

                                    violations.push(Violation {
                                        fact_id: target_key.clone(),
                                        fact_title: "Timeline: Knowledge before learning".to_string(),
                                        severity,
                                        message: format!(
                                            "Character knows about '{}' at scene '{}' (sequence {}) before learning it at event '{}' (sequence {})",
                                            target_key,
                                            scene.title,
                                            scene_event.sequence,
                                            learning_event.title,
                                            learning_sequence
                                        ),
                                        confidence: 0.9, // Sequence ordering is definitive
                                        auto_detected_as_intentional: false,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(violations)
    }

    /// Investigate all contradictions for an entity and its connected entities.
    ///
    /// Traverses the entity graph up to max_depth hops, checking each entity
    /// for fact, timeline, and relationship violations.
    pub async fn investigate_contradictions(
        &self,
        entity_id: &str,
        max_depth: usize,
    ) -> Result<(Vec<Violation>, usize), NarraError> {
        use std::collections::{HashSet, VecDeque};

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        let mut all_violations: Vec<Violation> = Vec::new();

        queue.push_back((entity_id.to_string(), 0));
        visited.insert(entity_id.to_string());

        while let Some((current_id, depth)) = queue.pop_front() {
            // Check this entity for violations
            let entity_type = current_id.split(':').next().unwrap_or("");

            // Fact violations
            let fact_result = self
                .check_entity_mutation(&current_id, &serde_json::json!({}))
                .await?;
            for (_sev, violations) in fact_result.violations_by_severity {
                all_violations.extend(violations);
            }

            // Timeline and relationship violations (only for characters)
            if entity_type == "character" {
                let char_id = current_id.split(':').nth(1).unwrap_or(&current_id);
                let timeline_v = self.check_timeline_violations(char_id).await?;
                let rel_v = self.check_relationship_violations(char_id).await?;
                all_violations.extend(timeline_v);
                all_violations.extend(rel_v);
            }

            // If not at max depth, find connected entities
            if depth < max_depth {
                let connected = self.get_connected_entities(&current_id).await?;
                for conn_id in connected {
                    if !visited.contains(&conn_id) {
                        visited.insert(conn_id.clone());
                        queue.push_back((conn_id, depth + 1));
                    }
                }
            }
        }

        Ok((all_violations, visited.len()))
    }

    /// Check for relationship violations for a character.
    ///
    /// Detects:
    /// - Impossible states: A is B's parent AND B is A's parent (CRITICAL)
    /// - Family/professional asymmetry: likely unintentional (WARNING)
    /// - Romantic/rival asymmetry: often intentional drama (INFO)
    ///
    /// Per CONTEXT.md:
    /// - One-way relationships are NOT violations (unrequited love valid)
    /// - Only flag asymmetry when reverse perception exists
    pub async fn check_relationship_violations(
        &self,
        character_id: &str,
    ) -> Result<Vec<Violation>, NarraError> {
        let mut violations = Vec::new();

        // Get all outgoing perceptions from this character (A->B)
        let outgoing_perceptions = get_perceptions_from(&self.db, character_id).await?;

        for perception_ab in &outgoing_perceptions {
            // Get the target character's key
            let char_b_key = perception_ab.to_character.key().to_string();

            // Get reverse perception B->A (if exists)
            if let Some(perception_ba) = get_perception(&self.db, &char_b_key, character_id).await?
            {
                // Check for impossible circular states
                // Both have "parent" in rel_types = circular parent
                if perception_ab.rel_types.contains(&"parent".to_string())
                    && perception_ba.rel_types.contains(&"parent".to_string())
                {
                    violations.push(Violation {
                        fact_id: format!("relationship:{}", perception_ab.id.key()),
                        fact_title: "Impossible relationship state".to_string(),
                        severity: ConsistencySeverity::Critical,
                        message: format!(
                            "Circular parent relationship: {} is parent of {} AND vice versa",
                            character_id, char_b_key
                        ),
                        confidence: 1.0, // Impossible states are definitive
                        auto_detected_as_intentional: false,
                    });
                }

                // Both have "child" in rel_types = circular child
                if perception_ab.rel_types.contains(&"child".to_string())
                    && perception_ba.rel_types.contains(&"child".to_string())
                {
                    violations.push(Violation {
                        fact_id: format!("relationship:{}", perception_ab.id.key()),
                        fact_title: "Impossible relationship state".to_string(),
                        severity: ConsistencySeverity::Critical,
                        message: format!(
                            "Circular child relationship: {} is child of {} AND vice versa",
                            character_id, char_b_key
                        ),
                        confidence: 1.0, // Impossible states are definitive
                        auto_detected_as_intentional: false,
                    });
                }

                // Check for asymmetric feelings (if both have feelings but differ)
                if let (Some(feelings_ab), Some(feelings_ba)) =
                    (&perception_ab.feelings, &perception_ba.feelings)
                {
                    if feelings_ab != feelings_ba {
                        // Determine severity based on relationship type
                        let has_family = perception_ab.rel_types.contains(&"family".to_string())
                            || perception_ba.rel_types.contains(&"family".to_string());
                        let has_professional = perception_ab
                            .rel_types
                            .contains(&"professional".to_string())
                            || perception_ba
                                .rel_types
                                .contains(&"professional".to_string());
                        let has_romantic =
                            perception_ab.rel_types.contains(&"romantic".to_string())
                                || perception_ba.rel_types.contains(&"romantic".to_string());
                        let has_rival = perception_ab.rel_types.contains(&"rivalry".to_string())
                            || perception_ba.rel_types.contains(&"rivalry".to_string());

                        let severity = if has_family || has_professional {
                            // Family/professional asymmetry likely unintentional
                            ConsistencySeverity::Warning
                        } else if has_romantic || has_rival {
                            // Romantic/rival asymmetry often intentional drama
                            ConsistencySeverity::Info
                        } else {
                            // Default to Info for other types
                            ConsistencySeverity::Info
                        };

                        let rel_type_str = if has_family {
                            "family"
                        } else if has_professional {
                            "professional"
                        } else if has_romantic {
                            "romantic"
                        } else if has_rival {
                            "rivalry"
                        } else {
                            "relationship"
                        };

                        violations.push(Violation {
                            fact_id: format!("relationship:{}", perception_ab.id.key()),
                            fact_title: "Relationship asymmetry".to_string(),
                            severity,
                            message: format!(
                                "Asymmetric {} relationship: {} feels '{}' but {} feels '{}'",
                                rel_type_str, character_id, feelings_ab, char_b_key, feelings_ba
                            ),
                            confidence: 0.6, // May be intentional
                            auto_detected_as_intentional: false,
                        });
                    }
                }
            }
            // If no reverse perception exists, that's valid (one-way relationships allowed)
        }

        Ok(violations)
    }

    /// Internal validation logic shared between mutation and creation checks.
    async fn perform_validation(
        &self,
        entity_id: &str,
        entity_data: &serde_json::Value,
    ) -> Result<ValidationResult, NarraError> {
        let mut result = ValidationResult::new();

        // Get applicable facts (linked facts or global Strict facts)
        let facts = self.get_applicable_facts(entity_id).await?;

        // Check for intentional contradiction pattern
        let is_intentional = self
            .is_intentional_contradiction(entity_id, entity_data)
            .await;

        // Pre-resolve current event sequence for temporal scope checks
        let current_event_sequence = self.get_latest_event_sequence().await;

        // Evaluate each fact against entity data
        for fact in facts {
            if let Some(violation) = self
                .evaluate_fact(
                    &fact,
                    entity_id,
                    entity_data,
                    is_intentional,
                    current_event_sequence,
                )
                .await
            {
                result.add_violation(violation);
            }
        }

        Ok(result)
    }
}

#[async_trait]
impl ConsistencyService for ConsistencyChecker {
    async fn check_entity_mutation(
        &self,
        entity_id: &str,
        mutation_fields: &serde_json::Value,
    ) -> Result<ValidationResult, NarraError> {
        // Use timeout to prevent long-running validation (2 second limit per RESEARCH.md)
        let validation = timeout(
            Duration::from_secs(2),
            self.perform_validation(entity_id, mutation_fields),
        )
        .await
        .map_err(|_| NarraError::Validation("Consistency check timeout".into()))?;

        validation
    }

    async fn check_entity_creation(
        &self,
        entity_type: &str,
        creation_data: &serde_json::Value,
    ) -> Result<ValidationResult, NarraError> {
        // For creation, construct a synthetic entity_id for fact lookup
        // We'll check against global Strict facts since entity doesn't exist yet
        let synthetic_id = format!("{}:__new__", entity_type);

        timeout(
            Duration::from_secs(2),
            self.perform_validation(&synthetic_id, creation_data),
        )
        .await
        .map_err(|_| NarraError::Validation("Consistency check timeout".into()))?
    }

    async fn check_timeline_violations(
        &self,
        character_id: &str,
    ) -> Result<Vec<Violation>, NarraError> {
        // Delegate to the inherent method
        self.check_timeline_violations(character_id).await
    }

    async fn check_relationship_violations(
        &self,
        character_id: &str,
    ) -> Result<Vec<Violation>, NarraError> {
        // Delegate to the inherent method
        self.check_relationship_violations(character_id).await
    }

    async fn investigate_contradictions(
        &self,
        entity_id: &str,
        max_depth: usize,
    ) -> Result<(Vec<Violation>, usize), NarraError> {
        // Delegate to the inherent method
        self.investigate_contradictions(entity_id, max_depth).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::fact::EnforcementLevel;

    fn make_violation(message: &str, severity: ConsistencySeverity) -> Violation {
        Violation {
            fact_id: "test:1".to_string(),
            fact_title: "Test fact".to_string(),
            severity,
            message: message.to_string(),
            confidence: 0.8,
            auto_detected_as_intentional: false,
        }
    }

    // -- generate_suggested_fix tests --

    #[test]
    fn test_suggested_fix_before_learning() {
        let v = make_violation(
            "Character knows X before learning it",
            ConsistencySeverity::Warning,
        );
        let fix = generate_suggested_fix(&v);
        assert!(fix.is_some());
        assert!(fix.unwrap().contains("Move the learning event"));
    }

    #[test]
    fn test_suggested_fix_circular_parent() {
        let v = make_violation(
            "Circular parent relationship detected",
            ConsistencySeverity::Critical,
        );
        let fix = generate_suggested_fix(&v);
        assert!(fix.is_some());
        assert!(fix.unwrap().contains("Remove one parent/child"));
    }

    #[test]
    fn test_suggested_fix_asymmetry() {
        let v = make_violation(
            "Asymmetric feelings between A and B",
            ConsistencySeverity::Info,
        );
        let fix = generate_suggested_fix(&v);
        assert!(fix.is_some());
        assert!(fix.unwrap().contains("intentional dramatic tension"));
    }

    #[test]
    fn test_suggested_fix_violate_fact() {
        let v = make_violation(
            "Entity may violate fact: No magic",
            ConsistencySeverity::Warning,
        );
        let fix = generate_suggested_fix(&v);
        assert!(fix.is_some());
        assert!(fix
            .unwrap()
            .contains("Review entity against the universe fact"));
    }

    #[test]
    fn test_suggested_fix_unknown_returns_none() {
        let v = make_violation("Some unknown issue", ConsistencySeverity::Info);
        assert!(generate_suggested_fix(&v).is_none());
    }

    // -- map_enforcement_to_severity tests --

    #[test]
    fn test_severity_strict_high_confidence_non_intentional() {
        assert_eq!(
            map_enforcement_to_severity(EnforcementLevel::Strict, 0.8, false),
            ConsistencySeverity::Critical
        );
    }

    #[test]
    fn test_severity_warning_high_confidence() {
        assert_eq!(
            map_enforcement_to_severity(EnforcementLevel::Warning, 0.8, false),
            ConsistencySeverity::Warning
        );
    }

    #[test]
    fn test_severity_intentional_always_info() {
        assert_eq!(
            map_enforcement_to_severity(EnforcementLevel::Strict, 0.9, true),
            ConsistencySeverity::Info
        );
    }

    #[test]
    fn test_severity_low_confidence_is_info() {
        assert_eq!(
            map_enforcement_to_severity(EnforcementLevel::Strict, 0.3, false),
            ConsistencySeverity::Info
        );
    }

    #[test]
    fn test_severity_informational_enforcement_is_info() {
        assert_eq!(
            map_enforcement_to_severity(EnforcementLevel::Informational, 0.9, false),
            ConsistencySeverity::Info
        );
    }

    // -- ValidationResult tests --

    #[test]
    fn test_validation_result_new_is_valid() {
        let result = ValidationResult::new();
        assert!(result.is_valid);
        assert_eq!(result.total_violations, 0);
        assert!(!result.has_blocking_violations);
    }

    #[test]
    fn test_add_critical_violation_invalidates() {
        let mut result = ValidationResult::new();
        result.add_violation(make_violation("critical", ConsistencySeverity::Critical));
        assert!(!result.is_valid);
        assert!(result.has_blocking_violations);
        assert_eq!(result.total_violations, 1);
    }

    #[test]
    fn test_add_warning_keeps_valid() {
        let mut result = ValidationResult::new();
        result.add_violation(make_violation("warning", ConsistencySeverity::Warning));
        assert!(result.is_valid);
        assert!(!result.has_blocking_violations);
        assert_eq!(result.total_violations, 1);
    }

    #[test]
    fn test_warnings_format() {
        let mut result = ValidationResult::new();
        result.add_violation(make_violation(
            "something wrong",
            ConsistencySeverity::Warning,
        ));
        result.add_violation(make_violation("fyi note", ConsistencySeverity::Info));

        let warnings = result.warnings();
        assert_eq!(warnings.len(), 2);
        assert!(warnings[0].starts_with("WARNING:"));
        assert!(warnings[1].starts_with("INFO:"));
    }
}
