//! Dramatic irony analysis service for knowledge asymmetry detection.
//!
//! Identifies situations where some characters know things others don't,
//! enabling writers to discover dramatic irony opportunities.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use surrealdb::RecordId;

use crate::db::connection::NarraDb;

use crate::models::fact::EnforcementLevel;
use crate::models::{Character, KnowledgeState};
use crate::NarraError;

/// Context data for a character pair from their perceives edge.
#[derive(Debug, Clone, Default)]
pub struct PairContext {
    pub tension_level: Option<i32>,
    pub feelings: Option<String>,
    pub history_notes: Option<String>,
}

/// A knowledge asymmetry between two characters.
#[derive(Debug, Clone, Serialize)]
pub struct KnowledgeAsymmetry {
    /// Character who knows (ID key only)
    pub knowing_character_id: String,
    /// Character who knows (name)
    pub knowing_character_name: String,
    /// Character who doesn't know (ID key only)
    pub unknowing_character_id: String,
    /// Character who doesn't know (name)
    pub unknowing_character_name: String,
    /// The fact that creates the asymmetry
    pub fact: String,
    /// How the knowing character learned
    pub learning_method: String,
    /// Certainty level of the knowing character
    pub certainty: String,
    /// Number of scenes since the asymmetry began
    pub scenes_since: usize,
    /// Signal strength: "high", "medium", "low"
    pub signal_strength: String,
    /// What/who the fact is about
    pub about: String,
    /// Weighted score incorporating tension and enforcement (higher = more dramatic)
    pub dramatic_weight: f32,
    /// Tension level from perceives edge between the two characters
    pub tension_level: Option<i32>,
    /// Feelings from perceives edge (observer → unknowing character)
    pub feelings: Option<String>,
    /// History notes from perceives edge
    pub history_notes: Option<String>,
}

/// Complete dramatic irony report.
#[derive(Debug, Clone, Serialize)]
pub struct IronyReport {
    /// Focus description (e.g., "All characters" or "Alice vs Bob")
    pub focus: String,
    /// Detected asymmetries
    pub asymmetries: Vec<KnowledgeAsymmetry>,
    /// Total count
    pub total_asymmetries: usize,
    /// Count of high-signal items
    pub high_signal_count: usize,
    /// Narrative opportunity suggestions
    pub narrative_opportunities: Vec<String>,
}

// ---------------------------------------------------------------------------
// Pure functions (testable without DB)
// ---------------------------------------------------------------------------

/// Classify the signal strength of a knowledge asymmetry.
pub(crate) fn classify_signal_strength(scenes_since: usize) -> &'static str {
    if scenes_since >= 5 {
        "high"
    } else if scenes_since >= 2 {
        "medium"
    } else {
        "low"
    }
}

/// Compute dramatic weight from scenes_since, tension_level, and enforcement level.
pub(crate) fn compute_dramatic_weight(
    scenes_since: usize,
    tension_level: Option<i32>,
    enforcement: Option<EnforcementLevel>,
) -> f32 {
    let mut weight = scenes_since as f32;
    if let Some(t) = tension_level {
        if t >= 7 {
            weight += 3.0;
        } else if t >= 4 {
            weight += 1.5;
        }
    }
    if let Some(e) = enforcement {
        match e {
            EnforcementLevel::Strict => weight += 3.0,
            EnforcementLevel::Warning => weight += 1.0,
            EnforcementLevel::Informational => {}
        }
    }
    weight
}

/// Generate narrative opportunity descriptions from high-signal asymmetries.
pub(crate) fn generate_narrative_opportunities(
    asymmetries: &[KnowledgeAsymmetry],
    limit: usize,
) -> Vec<String> {
    asymmetries
        .iter()
        .filter(|a| a.signal_strength == "high")
        .take(limit)
        .map(|a| {
            let mut desc = format!(
                "{} doesn't know {} ({} scenes since {} learned)",
                a.unknowing_character_name, a.fact, a.scenes_since, a.knowing_character_name
            );
            if let Some(ref feelings) = a.feelings {
                desc.push_str(&format!(" [feels: {}]", feelings));
            }
            if let Some(ref history) = a.history_notes {
                desc.push_str(&format!(" [history: {}]", history));
            }
            desc
        })
        .collect()
}

/// Sort asymmetries by dramatic_weight descending, then signal strength, then scenes_since.
pub(crate) fn sort_asymmetries(asymmetries: &mut [KnowledgeAsymmetry]) {
    asymmetries.sort_by(|a, b| {
        // Primary: dramatic_weight descending
        b.dramatic_weight
            .partial_cmp(&a.dramatic_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                // Secondary: signal_strength bucket
                let strength_ord = |s: &str| -> u8 {
                    match s {
                        "high" => 0,
                        "medium" => 1,
                        _ => 2,
                    }
                };
                strength_ord(a.signal_strength.as_str())
                    .cmp(&strength_ord(b.signal_strength.as_str()))
            })
            .then_with(|| {
                // Tertiary: scenes_since descending
                b.scenes_since.cmp(&a.scenes_since)
            })
    });
}

// ---------------------------------------------------------------------------
// Data provider trait (enables mock-based testing)
// ---------------------------------------------------------------------------

/// Data access abstraction for the irony service.
#[async_trait]
pub trait IronyDataProvider: Send + Sync {
    async fn get_scene_count(&self) -> Result<usize, NarraError>;
    async fn get_all_characters(&self) -> Result<Vec<Character>, NarraError>;
    async fn get_character(&self, id: &str) -> Result<Option<Character>, NarraError>;
    async fn get_character_knowledge(
        &self,
        character_id: &str,
    ) -> Result<Vec<KnowledgeState>, NarraError>;
    async fn get_fact_text(&self, target: &RecordId) -> Result<String, NarraError>;
    async fn estimate_scenes_since(
        &self,
        learned_at: &surrealdb::Datetime,
    ) -> Result<usize, NarraError>;

    /// Bulk-fetch all fact texts (knowledge facts + character names).
    /// Returns a map from target RecordId string to display text.
    async fn get_all_fact_texts(&self) -> Result<HashMap<String, String>, NarraError>;

    /// Get all scene created_at timestamps in descending order.
    /// Used to compute scenes_since without per-item queries.
    async fn get_scene_timestamps(&self) -> Result<Vec<surrealdb::Datetime>, NarraError>;

    /// Get tension/feelings/history for all perceives edges with tension_level set.
    /// Returns `(from_char_key, to_char_key) -> PairContext`.
    async fn get_all_perception_contexts(
        &self,
    ) -> Result<HashMap<(String, String), PairContext>, NarraError>;

    /// Get enforcement level for knowledge targets via knows → target linkage.
    /// Returns `knowledge_target_id -> EnforcementLevel` for targets linked to facts.
    async fn get_fact_enforcement_levels(
        &self,
    ) -> Result<HashMap<String, EnforcementLevel>, NarraError>;
}

/// SurrealDB implementation of IronyDataProvider.
pub struct SurrealIronyDataProvider {
    db: Arc<NarraDb>,
}

impl SurrealIronyDataProvider {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl IronyDataProvider for SurrealIronyDataProvider {
    async fn get_scene_count(&self) -> Result<usize, NarraError> {
        let mut result = self.db.query("SELECT count() FROM scene GROUP ALL").await?;

        #[derive(Deserialize)]
        struct CountResult {
            count: usize,
        }

        let counts: Vec<CountResult> = result.take(0)?;
        Ok(counts.first().map(|c| c.count).unwrap_or(0))
    }

    async fn get_all_characters(&self) -> Result<Vec<Character>, NarraError> {
        let characters: Vec<Character> = self.db.select("character").await?;
        Ok(characters)
    }

    async fn get_character(&self, id: &str) -> Result<Option<Character>, NarraError> {
        let result: Option<Character> = self.db.select(("character", id)).await?;
        Ok(result)
    }

    async fn get_character_knowledge(
        &self,
        character_id: &str,
    ) -> Result<Vec<KnowledgeState>, NarraError> {
        let query = format!(
            "SELECT * FROM knows WHERE in = character:{} ORDER BY learned_at DESC LIMIT 100",
            character_id
        );
        let mut result = self.db.query(&query).await?;
        let states: Vec<KnowledgeState> = result.take(0)?;
        Ok(states)
    }

    async fn get_fact_text(&self, target: &RecordId) -> Result<String, NarraError> {
        let target_str = target.to_string();
        let key = target.key().to_string();
        let table = target_str.split(':').next().unwrap_or("unknown");

        match table {
            "knowledge" => {
                let query = format!("SELECT fact FROM knowledge:{}", key);
                let mut result = self.db.query(&query).await?;

                #[derive(Deserialize)]
                struct FactResult {
                    fact: String,
                }

                let facts: Vec<FactResult> = result.take(0)?;
                Ok(facts
                    .first()
                    .map(|f| f.fact.clone())
                    .unwrap_or_else(|| format!("knowledge:{}", key)))
            }
            "character" => {
                let query = format!("SELECT name FROM character:{}", key);
                let mut result = self.db.query(&query).await?;

                #[derive(Deserialize)]
                struct NameResult {
                    name: String,
                }

                let names: Vec<NameResult> = result.take(0)?;
                Ok(names
                    .first()
                    .map(|n| format!("knows about {}", n.name))
                    .unwrap_or_else(|| format!("character:{}", key)))
            }
            _ => Ok(target.to_string()),
        }
    }

    async fn estimate_scenes_since(
        &self,
        learned_at: &surrealdb::Datetime,
    ) -> Result<usize, NarraError> {
        let learned_at_copy = learned_at.clone();
        let query = "SELECT count() FROM scene WHERE created_at > $learned_at GROUP ALL";
        let mut result = self
            .db
            .query(query)
            .bind(("learned_at", learned_at_copy))
            .await?;

        #[derive(Deserialize)]
        struct CountResult {
            count: usize,
        }

        let counts: Vec<CountResult> = result.take(0)?;
        Ok(counts.first().map(|c| c.count).unwrap_or(0))
    }

    async fn get_all_fact_texts(&self) -> Result<HashMap<String, String>, NarraError> {
        let mut map = HashMap::new();

        // Fetch all knowledge facts
        let mut result = self.db.query("SELECT id, fact FROM knowledge").await?;

        #[derive(Deserialize)]
        struct KnowledgeFact {
            id: surrealdb::RecordId,
            fact: String,
        }

        let facts: Vec<KnowledgeFact> = result.take(0)?;
        for f in facts {
            map.insert(f.id.to_string(), f.fact);
        }

        // Fetch all character names (for "knows about <name>" display)
        let mut result = self.db.query("SELECT id, name FROM character").await?;

        #[derive(Deserialize)]
        struct CharName {
            id: surrealdb::RecordId,
            name: String,
        }

        let chars: Vec<CharName> = result.take(0)?;
        for c in chars {
            map.insert(c.id.to_string(), format!("knows about {}", c.name));
        }

        Ok(map)
    }

    async fn get_scene_timestamps(&self) -> Result<Vec<surrealdb::Datetime>, NarraError> {
        let mut result = self
            .db
            .query("SELECT VALUE created_at FROM scene ORDER BY created_at DESC")
            .await?;
        let timestamps: Vec<surrealdb::Datetime> = result.take(0)?;
        Ok(timestamps)
    }

    async fn get_all_perception_contexts(
        &self,
    ) -> Result<HashMap<(String, String), PairContext>, NarraError> {
        let mut result = self
            .db
            .query(
                "SELECT in, out, tension_level, feelings, history_notes \
                 FROM perceives \
                 WHERE tension_level IS NOT NONE OR feelings IS NOT NONE OR history_notes IS NOT NONE",
            )
            .await?;

        #[derive(Deserialize)]
        struct PerceptionRow {
            #[serde(rename = "in")]
            from: surrealdb::RecordId,
            #[serde(rename = "out")]
            to: surrealdb::RecordId,
            tension_level: Option<i32>,
            feelings: Option<String>,
            history_notes: Option<String>,
        }

        let rows: Vec<PerceptionRow> = result.take(0)?;
        let mut map = HashMap::new();
        for row in rows {
            let from_key = row.from.key().to_string();
            let to_key = row.to.key().to_string();
            map.insert(
                (from_key, to_key),
                PairContext {
                    tension_level: row.tension_level,
                    feelings: row.feelings,
                    history_notes: row.history_notes,
                },
            );
        }
        Ok(map)
    }

    async fn get_fact_enforcement_levels(
        &self,
    ) -> Result<HashMap<String, EnforcementLevel>, NarraError> {
        // Join knowledge targets through applies_to to universe_fact
        let mut result = self
            .db
            .query(
                "SELECT out AS target, in.enforcement_level AS enforcement_level \
                 FROM applies_to \
                 WHERE in.tb = 'universe_fact'",
            )
            .await?;

        #[derive(Deserialize)]
        struct EnforcementRow {
            target: surrealdb::RecordId,
            enforcement_level: Option<EnforcementLevel>,
        }

        let rows: Vec<EnforcementRow> = result.take(0)?;
        let mut map = HashMap::new();
        for row in rows {
            if let Some(level) = row.enforcement_level {
                map.insert(row.target.to_string(), level);
            }
        }
        Ok(map)
    }
}

// ---------------------------------------------------------------------------
// IronyService (uses data provider)
// ---------------------------------------------------------------------------

/// Dramatic irony analysis service.
pub struct IronyService {
    data: Arc<dyn IronyDataProvider>,
}

impl IronyService {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self {
            data: Arc::new(SurrealIronyDataProvider::new(db)),
        }
    }

    pub fn with_provider(data: Arc<dyn IronyDataProvider>) -> Self {
        Self { data }
    }

    /// Generate a dramatic irony report.
    pub async fn generate_report(
        &self,
        character_id: Option<&str>,
        min_scene_threshold: usize,
    ) -> Result<IronyReport, NarraError> {
        let characters = self.data.get_all_characters().await?;

        let char_names: HashMap<String, String> = characters
            .iter()
            .map(|c| (c.id.key().to_string(), c.name.clone()))
            .collect();

        // Bulk pre-fetch: all fact texts, scene timestamps, tensions, enforcement (eliminates N+1)
        let fact_texts = self.data.get_all_fact_texts().await?;
        let scene_timestamps = self.data.get_scene_timestamps().await?;
        let perception_contexts = self.data.get_all_perception_contexts().await?;
        let enforcement_levels = self.data.get_fact_enforcement_levels().await?;

        // Build knowledge map
        let mut character_knowledge: HashMap<String, Vec<KnowledgeState>> = HashMap::new();
        for character in &characters {
            let char_id = character.id.key().to_string();
            let knowledge = self.data.get_character_knowledge(&char_id).await?;
            character_knowledge.insert(char_id, knowledge);
        }

        let mut asymmetries = Vec::new();

        let chars_to_compare: Vec<String> = match character_id {
            Some(cid) => vec![cid.to_string()],
            None => characters.iter().map(|c| c.id.key().to_string()).collect(),
        };

        let empty_vec = Vec::new();

        for char_a in &chars_to_compare {
            let a_knowledge = character_knowledge.get(char_a).unwrap_or(&empty_vec);

            for char_b in characters.iter().map(|c| c.id.key().to_string()) {
                if char_a == &char_b {
                    continue;
                }

                let b_knowledge = character_knowledge.get(&char_b).unwrap_or(&empty_vec);

                let b_targets: HashSet<String> =
                    b_knowledge.iter().map(|k| k.target.to_string()).collect();

                for a_knows in a_knowledge {
                    let target_str = a_knows.target.to_string();

                    if b_targets.contains(&target_str) {
                        continue;
                    }

                    // Use pre-fetched fact text (O(1) lookup instead of DB query)
                    let fact = fact_texts
                        .get(&target_str)
                        .cloned()
                        .unwrap_or_else(|| target_str.clone());

                    // Compute scenes_since from pre-fetched timestamps (O(log N) binary search)
                    let scenes_since =
                        scene_timestamps.partition_point(|ts| *ts > a_knows.learned_at);

                    if scenes_since < min_scene_threshold {
                        continue;
                    }

                    // Look up pair context (knowing → unknowing direction)
                    let pair_ctx = perception_contexts
                        .get(&(char_a.clone(), char_b.clone()))
                        .cloned()
                        .unwrap_or_default();

                    // Look up enforcement level for the knowledge target
                    let enforcement = enforcement_levels.get(&target_str).copied();

                    let dramatic_weight =
                        compute_dramatic_weight(scenes_since, pair_ctx.tension_level, enforcement);

                    asymmetries.push(KnowledgeAsymmetry {
                        knowing_character_id: char_a.clone(),
                        knowing_character_name: char_names.get(char_a).cloned().unwrap_or_default(),
                        unknowing_character_id: char_b.clone(),
                        unknowing_character_name: char_names
                            .get(&char_b)
                            .cloned()
                            .unwrap_or_default(),
                        fact,
                        learning_method: format!("{:?}", a_knows.learning_method),
                        certainty: format!("{:?}", a_knows.certainty),
                        scenes_since,
                        signal_strength: classify_signal_strength(scenes_since).to_string(),
                        about: target_str,
                        dramatic_weight,
                        tension_level: pair_ctx.tension_level,
                        feelings: pair_ctx.feelings,
                        history_notes: pair_ctx.history_notes,
                    });
                }
            }
        }

        sort_asymmetries(&mut asymmetries);

        let high_signal_count = asymmetries
            .iter()
            .filter(|a| a.signal_strength == "high")
            .count();

        let narrative_opportunities = generate_narrative_opportunities(&asymmetries, 5);

        let focus = match character_id {
            Some(cid) => format!(
                "Character: {}",
                char_names.get(cid).unwrap_or(&cid.to_string())
            ),
            None => "All characters".to_string(),
        };

        Ok(IronyReport {
            focus,
            total_asymmetries: asymmetries.len(),
            high_signal_count,
            asymmetries,
            narrative_opportunities,
        })
    }

    /// Detect asymmetries between two specific characters (bidirectional).
    /// Enriched with tension/enforcement context for accurate dramatic weighting.
    pub async fn detect_asymmetries(
        &self,
        character_a: &str,
        character_b: &str,
    ) -> Result<Vec<KnowledgeAsymmetry>, NarraError> {
        let char_a = self.data.get_character(character_a).await?;
        let char_b = self.data.get_character(character_b).await?;

        let char_a = char_a.ok_or_else(|| NarraError::NotFound {
            entity_type: "character".to_string(),
            id: character_a.to_string(),
        })?;
        let char_b = char_b.ok_or_else(|| NarraError::NotFound {
            entity_type: "character".to_string(),
            id: character_b.to_string(),
        })?;

        let a_knowledge = self.data.get_character_knowledge(character_a).await?;
        let b_knowledge = self.data.get_character_knowledge(character_b).await?;

        // Bulk-fetch perception contexts and enforcement levels
        let perception_contexts = self.data.get_all_perception_contexts().await?;
        let enforcement_levels = self.data.get_fact_enforcement_levels().await?;

        let mut asymmetries = Vec::new();

        // A knows, B doesn't
        let b_targets: HashSet<String> = b_knowledge.iter().map(|k| k.target.to_string()).collect();

        for a_knows in &a_knowledge {
            let target_str = a_knows.target.to_string();
            if b_targets.contains(&target_str) {
                continue;
            }

            let fact = self.data.get_fact_text(&a_knows.target).await?;
            let scenes_since = self.data.estimate_scenes_since(&a_knows.learned_at).await?;

            // Look up pair context (knowing → unknowing direction)
            let pair_ctx = perception_contexts
                .get(&(character_a.to_string(), character_b.to_string()))
                .cloned()
                .unwrap_or_default();
            let enforcement = enforcement_levels.get(&target_str).copied();

            asymmetries.push(KnowledgeAsymmetry {
                knowing_character_id: character_a.to_string(),
                knowing_character_name: char_a.name.clone(),
                unknowing_character_id: character_b.to_string(),
                unknowing_character_name: char_b.name.clone(),
                fact: fact.clone(),
                learning_method: format!("{:?}", a_knows.learning_method),
                certainty: format!("{:?}", a_knows.certainty),
                scenes_since,
                signal_strength: classify_signal_strength(scenes_since).to_string(),
                about: target_str,
                dramatic_weight: compute_dramatic_weight(
                    scenes_since,
                    pair_ctx.tension_level,
                    enforcement,
                ),
                tension_level: pair_ctx.tension_level,
                feelings: pair_ctx.feelings,
                history_notes: pair_ctx.history_notes,
            });
        }

        // B knows, A doesn't
        let a_targets: HashSet<String> = a_knowledge.iter().map(|k| k.target.to_string()).collect();

        for b_knows in &b_knowledge {
            let target_str = b_knows.target.to_string();
            if a_targets.contains(&target_str) {
                continue;
            }

            let fact = self.data.get_fact_text(&b_knows.target).await?;
            let scenes_since = self.data.estimate_scenes_since(&b_knows.learned_at).await?;

            // Look up pair context (knowing → unknowing direction)
            let pair_ctx = perception_contexts
                .get(&(character_b.to_string(), character_a.to_string()))
                .cloned()
                .unwrap_or_default();
            let enforcement = enforcement_levels.get(&target_str).copied();

            asymmetries.push(KnowledgeAsymmetry {
                knowing_character_id: character_b.to_string(),
                knowing_character_name: char_b.name.clone(),
                unknowing_character_id: character_a.to_string(),
                unknowing_character_name: char_a.name.clone(),
                fact: fact.clone(),
                learning_method: format!("{:?}", b_knows.learning_method),
                certainty: format!("{:?}", b_knows.certainty),
                scenes_since,
                signal_strength: classify_signal_strength(scenes_since).to_string(),
                about: target_str,
                dramatic_weight: compute_dramatic_weight(
                    scenes_since,
                    pair_ctx.tension_level,
                    enforcement,
                ),
                tension_level: pair_ctx.tension_level,
                feelings: pair_ctx.feelings,
                history_notes: pair_ctx.history_notes,
            });
        }

        sort_asymmetries(&mut asymmetries);

        Ok(asymmetries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::fact::EnforcementLevel;

    fn make_asymmetry(
        knowing: &str,
        unknowing: &str,
        fact: &str,
        scenes_since: usize,
        signal: &str,
    ) -> KnowledgeAsymmetry {
        KnowledgeAsymmetry {
            knowing_character_id: knowing.to_string(),
            knowing_character_name: knowing.to_string(),
            unknowing_character_id: unknowing.to_string(),
            unknowing_character_name: unknowing.to_string(),
            fact: fact.to_string(),
            learning_method: "witnessed".to_string(),
            certainty: "knows".to_string(),
            scenes_since,
            signal_strength: signal.to_string(),
            about: "target".to_string(),
            dramatic_weight: compute_dramatic_weight(scenes_since, None, None),
            tension_level: None,
            feelings: None,
            history_notes: None,
        }
    }

    // -- Pure function tests --

    #[test]
    fn test_classify_signal_strength_high() {
        assert_eq!(classify_signal_strength(5), "high");
        assert_eq!(classify_signal_strength(10), "high");
        assert_eq!(classify_signal_strength(100), "high");
    }

    #[test]
    fn test_classify_signal_strength_medium() {
        assert_eq!(classify_signal_strength(2), "medium");
        assert_eq!(classify_signal_strength(3), "medium");
        assert_eq!(classify_signal_strength(4), "medium");
    }

    #[test]
    fn test_classify_signal_strength_low() {
        assert_eq!(classify_signal_strength(0), "low");
        assert_eq!(classify_signal_strength(1), "low");
    }

    #[test]
    fn test_sort_asymmetries_by_signal_then_scenes() {
        let mut asymmetries = vec![
            make_asymmetry("Alice", "Bob", "secret1", 1, "low"),
            make_asymmetry("Alice", "Bob", "secret2", 10, "high"),
            make_asymmetry("Alice", "Bob", "secret3", 3, "medium"),
            make_asymmetry("Alice", "Bob", "secret4", 7, "high"),
        ];

        sort_asymmetries(&mut asymmetries);

        assert_eq!(asymmetries[0].signal_strength, "high");
        assert_eq!(asymmetries[0].scenes_since, 10);
        assert_eq!(asymmetries[1].signal_strength, "high");
        assert_eq!(asymmetries[1].scenes_since, 7);
        assert_eq!(asymmetries[2].signal_strength, "medium");
        assert_eq!(asymmetries[3].signal_strength, "low");
    }

    #[test]
    fn test_sort_asymmetries_empty() {
        let mut asymmetries: Vec<KnowledgeAsymmetry> = vec![];
        sort_asymmetries(&mut asymmetries);
        assert!(asymmetries.is_empty());
    }

    #[test]
    fn test_generate_narrative_opportunities_filters_high() {
        let asymmetries = vec![
            make_asymmetry("Alice", "Bob", "the treasure location", 10, "high"),
            make_asymmetry("Alice", "Charlie", "the secret door", 3, "medium"),
            make_asymmetry("Bob", "Charlie", "the betrayal plan", 8, "high"),
        ];

        let opportunities = generate_narrative_opportunities(&asymmetries, 5);
        assert_eq!(
            opportunities.len(),
            2,
            "Should only include high-signal items"
        );
        assert!(opportunities[0].contains("Bob"));
        assert!(opportunities[0].contains("the treasure location"));
        assert!(opportunities[1].contains("Charlie"));
    }

    #[test]
    fn test_generate_narrative_opportunities_respects_limit() {
        let asymmetries = vec![
            make_asymmetry("Alice", "Bob", "fact1", 10, "high"),
            make_asymmetry("Alice", "Charlie", "fact2", 8, "high"),
            make_asymmetry("Alice", "Dave", "fact3", 6, "high"),
        ];

        let opportunities = generate_narrative_opportunities(&asymmetries, 2);
        assert_eq!(opportunities.len(), 2);
    }

    #[test]
    fn test_generate_narrative_opportunities_empty() {
        let opportunities = generate_narrative_opportunities(&[], 5);
        assert!(opportunities.is_empty());
    }

    // -- Mock-based service tests --

    struct MockIronyDataProvider {
        characters: Vec<Character>,
        knowledge: HashMap<String, Vec<KnowledgeState>>,
        scene_count: usize,
    }

    #[async_trait]
    impl IronyDataProvider for MockIronyDataProvider {
        async fn get_scene_count(&self) -> Result<usize, NarraError> {
            Ok(self.scene_count)
        }

        async fn get_all_characters(&self) -> Result<Vec<Character>, NarraError> {
            Ok(self.characters.clone())
        }

        async fn get_character(&self, id: &str) -> Result<Option<Character>, NarraError> {
            Ok(self
                .characters
                .iter()
                .find(|c| c.id.key().to_string() == id)
                .cloned())
        }

        async fn get_character_knowledge(
            &self,
            character_id: &str,
        ) -> Result<Vec<KnowledgeState>, NarraError> {
            Ok(self
                .knowledge
                .get(character_id)
                .cloned()
                .unwrap_or_default())
        }

        async fn get_fact_text(&self, target: &RecordId) -> Result<String, NarraError> {
            Ok(format!("fact about {}", target))
        }

        async fn estimate_scenes_since(
            &self,
            _learned_at: &surrealdb::Datetime,
        ) -> Result<usize, NarraError> {
            // Return scene_count as a simple estimate
            Ok(self.scene_count)
        }

        async fn get_all_fact_texts(&self) -> Result<HashMap<String, String>, NarraError> {
            // Build map from all knowledge targets referenced in mock data
            let mut map = HashMap::new();
            for states in self.knowledge.values() {
                for k in states {
                    map.insert(k.target.to_string(), format!("fact about {}", k.target));
                }
            }
            Ok(map)
        }

        async fn get_scene_timestamps(&self) -> Result<Vec<surrealdb::Datetime>, NarraError> {
            // Return scene_count timestamps spaced 1 second apart (descending)
            let base = chrono::Utc::now();
            let timestamps: Vec<surrealdb::Datetime> = (0..self.scene_count)
                .map(|i| {
                    let t = base - chrono::Duration::seconds(i as i64);
                    surrealdb::Datetime::from(t)
                })
                .collect();
            Ok(timestamps)
        }

        async fn get_all_perception_contexts(
            &self,
        ) -> Result<HashMap<(String, String), PairContext>, NarraError> {
            Ok(HashMap::new())
        }

        async fn get_fact_enforcement_levels(
            &self,
        ) -> Result<HashMap<String, EnforcementLevel>, NarraError> {
            Ok(HashMap::new())
        }
    }

    fn mock_character(id: &str, name: &str) -> Character {
        use surrealdb::RecordId;
        Character {
            id: RecordId::from(("character", id)),
            name: name.to_string(),
            aliases: vec![],
            roles: vec![],
            profile: std::collections::HashMap::new(),
            created_at: Default::default(),
            updated_at: Default::default(),
        }
    }

    fn mock_knowledge(target_table: &str, target_id: &str) -> KnowledgeState {
        use crate::models::{CertaintyLevel, LearningMethod};
        use surrealdb::RecordId;
        // Use a fixed time in the past so scene_timestamps (generated near "now")
        // are clearly after learned_at.
        let past = chrono::Utc::now() - chrono::Duration::hours(1);
        KnowledgeState {
            id: RecordId::from(("knows", "test")),
            character: RecordId::from(("character", "test")),
            target: RecordId::from((target_table, target_id)),
            certainty: CertaintyLevel::Knows,
            learning_method: LearningMethod::Witnessed,
            source_character: None,
            event: None,
            premises: None,
            truth_value: None,
            learned_at: surrealdb::Datetime::from(past),
            created_at: Default::default(),
            updated_at: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_irony_service_no_characters() {
        let provider = MockIronyDataProvider {
            characters: vec![],
            knowledge: HashMap::new(),
            scene_count: 10,
        };
        let service = IronyService::with_provider(Arc::new(provider));

        let report = service.generate_report(None, 0).await.unwrap();
        assert_eq!(report.total_asymmetries, 0);
        assert_eq!(report.focus, "All characters");
    }

    #[tokio::test]
    async fn test_irony_service_single_character() {
        let provider = MockIronyDataProvider {
            characters: vec![mock_character("alice", "Alice")],
            knowledge: HashMap::new(),
            scene_count: 10,
        };
        let service = IronyService::with_provider(Arc::new(provider));

        let report = service.generate_report(None, 0).await.unwrap();
        assert_eq!(report.total_asymmetries, 0);
    }

    #[tokio::test]
    async fn test_irony_service_detects_asymmetry() {
        let mut knowledge = HashMap::new();
        // Alice knows something Bob doesn't
        knowledge.insert(
            "alice".to_string(),
            vec![mock_knowledge("knowledge", "secret_treasure")],
        );
        knowledge.insert("bob".to_string(), vec![]);

        let provider = MockIronyDataProvider {
            characters: vec![
                mock_character("alice", "Alice"),
                mock_character("bob", "Bob"),
            ],
            knowledge,
            scene_count: 10, // Will make signal "high" (>= 5)
        };
        let service = IronyService::with_provider(Arc::new(provider));

        let report = service.generate_report(None, 0).await.unwrap();
        assert_eq!(report.total_asymmetries, 1);
        assert_eq!(report.high_signal_count, 1);
        assert_eq!(report.asymmetries[0].knowing_character_name, "Alice");
        assert_eq!(report.asymmetries[0].unknowing_character_name, "Bob");
        assert_eq!(report.asymmetries[0].signal_strength, "high");
    }

    #[tokio::test]
    async fn test_irony_service_bidirectional_asymmetry() {
        let mut knowledge = HashMap::new();
        // Alice knows X, Bob knows Y — each has something the other doesn't
        knowledge.insert(
            "alice".to_string(),
            vec![mock_knowledge("knowledge", "x_secret")],
        );
        knowledge.insert(
            "bob".to_string(),
            vec![mock_knowledge("knowledge", "y_secret")],
        );

        let provider = MockIronyDataProvider {
            characters: vec![
                mock_character("alice", "Alice"),
                mock_character("bob", "Bob"),
            ],
            knowledge,
            scene_count: 10,
        };
        let service = IronyService::with_provider(Arc::new(provider));

        let report = service.generate_report(None, 0).await.unwrap();
        assert_eq!(
            report.total_asymmetries, 2,
            "Each knows something the other doesn't"
        );
    }

    #[tokio::test]
    async fn test_irony_service_shared_knowledge_no_asymmetry() {
        let shared = mock_knowledge("knowledge", "common_fact");
        let mut knowledge = HashMap::new();
        knowledge.insert("alice".to_string(), vec![shared.clone()]);
        knowledge.insert("bob".to_string(), vec![shared]);

        let provider = MockIronyDataProvider {
            characters: vec![
                mock_character("alice", "Alice"),
                mock_character("bob", "Bob"),
            ],
            knowledge,
            scene_count: 10,
        };
        let service = IronyService::with_provider(Arc::new(provider));

        let report = service.generate_report(None, 0).await.unwrap();
        assert_eq!(
            report.total_asymmetries, 0,
            "No asymmetry when both know the same fact"
        );
    }

    #[test]
    fn test_compute_dramatic_weight_basic() {
        // Base: just scenes_since
        assert_eq!(compute_dramatic_weight(5, None, None), 5.0);
        // High tension adds 3.0
        assert_eq!(compute_dramatic_weight(5, Some(8), None), 8.0);
        // Medium tension adds 1.5
        assert_eq!(compute_dramatic_weight(5, Some(5), None), 6.5);
        // Low tension adds nothing
        assert_eq!(compute_dramatic_weight(5, Some(2), None), 5.0);
        // Strict enforcement adds 3.0
        assert_eq!(
            compute_dramatic_weight(5, None, Some(EnforcementLevel::Strict)),
            8.0
        );
        // Warning enforcement adds 1.0
        assert_eq!(
            compute_dramatic_weight(5, None, Some(EnforcementLevel::Warning)),
            6.0
        );
        // Combined: high tension + strict
        assert_eq!(
            compute_dramatic_weight(5, Some(9), Some(EnforcementLevel::Strict)),
            11.0
        );
    }

    #[test]
    fn test_sort_asymmetries_uses_dramatic_weight() {
        let mut asymmetries = vec![
            {
                let mut a = make_asymmetry("Alice", "Bob", "low_weight", 6, "high");
                a.dramatic_weight = 6.0; // just scenes_since
                a
            },
            {
                let mut a = make_asymmetry("Alice", "Charlie", "high_weight", 6, "high");
                a.dramatic_weight = 12.0; // tension + enforcement boosted
                a
            },
        ];

        sort_asymmetries(&mut asymmetries);

        // Higher dramatic_weight should sort first
        assert_eq!(asymmetries[0].fact, "high_weight");
        assert_eq!(asymmetries[1].fact, "low_weight");
    }

    // -- Mock with tension + enforcement support --

    /// Extended mock provider that returns tension contexts and enforcement levels.
    struct MockIronyDataProviderWithContext {
        characters: Vec<Character>,
        knowledge: HashMap<String, Vec<KnowledgeState>>,
        scene_count: usize,
        perception_contexts: HashMap<(String, String), PairContext>,
        enforcement_levels: HashMap<String, EnforcementLevel>,
    }

    #[async_trait]
    impl IronyDataProvider for MockIronyDataProviderWithContext {
        async fn get_scene_count(&self) -> Result<usize, NarraError> {
            Ok(self.scene_count)
        }

        async fn get_all_characters(&self) -> Result<Vec<Character>, NarraError> {
            Ok(self.characters.clone())
        }

        async fn get_character(&self, id: &str) -> Result<Option<Character>, NarraError> {
            Ok(self
                .characters
                .iter()
                .find(|c| c.id.key().to_string() == id)
                .cloned())
        }

        async fn get_character_knowledge(
            &self,
            character_id: &str,
        ) -> Result<Vec<KnowledgeState>, NarraError> {
            Ok(self
                .knowledge
                .get(character_id)
                .cloned()
                .unwrap_or_default())
        }

        async fn get_fact_text(&self, target: &RecordId) -> Result<String, NarraError> {
            Ok(format!("fact about {}", target))
        }

        async fn estimate_scenes_since(
            &self,
            _learned_at: &surrealdb::Datetime,
        ) -> Result<usize, NarraError> {
            Ok(self.scene_count)
        }

        async fn get_all_fact_texts(&self) -> Result<HashMap<String, String>, NarraError> {
            let mut map = HashMap::new();
            for states in self.knowledge.values() {
                for k in states {
                    map.insert(k.target.to_string(), format!("fact about {}", k.target));
                }
            }
            Ok(map)
        }

        async fn get_scene_timestamps(&self) -> Result<Vec<surrealdb::Datetime>, NarraError> {
            let base = chrono::Utc::now();
            let timestamps: Vec<surrealdb::Datetime> = (0..self.scene_count)
                .map(|i| {
                    let t = base - chrono::Duration::seconds(i as i64);
                    surrealdb::Datetime::from(t)
                })
                .collect();
            Ok(timestamps)
        }

        async fn get_all_perception_contexts(
            &self,
        ) -> Result<HashMap<(String, String), PairContext>, NarraError> {
            Ok(self.perception_contexts.clone())
        }

        async fn get_fact_enforcement_levels(
            &self,
        ) -> Result<HashMap<String, EnforcementLevel>, NarraError> {
            Ok(self.enforcement_levels.clone())
        }
    }

    #[tokio::test]
    async fn test_irony_report_includes_tension_from_perception_context() {
        let mut knowledge = HashMap::new();
        // Alice knows something Bob doesn't
        knowledge.insert(
            "alice".to_string(),
            vec![mock_knowledge("knowledge", "the_secret")],
        );
        knowledge.insert("bob".to_string(), vec![]);

        // Alice → Bob has high tension
        let mut perception_contexts = HashMap::new();
        perception_contexts.insert(
            ("alice".to_string(), "bob".to_string()),
            PairContext {
                tension_level: Some(9),
                feelings: Some("bitter resentment".to_string()),
                history_notes: Some("They were once allies".to_string()),
            },
        );

        let provider = MockIronyDataProviderWithContext {
            characters: vec![
                mock_character("alice", "Alice"),
                mock_character("bob", "Bob"),
            ],
            knowledge,
            scene_count: 10,
            perception_contexts,
            enforcement_levels: HashMap::new(),
        };
        let service = IronyService::with_provider(Arc::new(provider));

        let report = service.generate_report(None, 0).await.unwrap();
        assert_eq!(report.total_asymmetries, 1);

        let asym = &report.asymmetries[0];
        assert_eq!(asym.tension_level, Some(9), "Should carry tension_level");
        assert_eq!(
            asym.feelings.as_deref(),
            Some("bitter resentment"),
            "Should carry feelings"
        );
        assert_eq!(
            asym.history_notes.as_deref(),
            Some("They were once allies"),
            "Should carry history_notes"
        );
        // dramatic_weight = 10 (scenes_since) + 3.0 (tension >= 7) = 13.0
        assert!(
            (asym.dramatic_weight - 13.0).abs() < 0.01,
            "dramatic_weight should be 13.0, got {}",
            asym.dramatic_weight
        );
    }

    #[tokio::test]
    async fn test_irony_report_enforcement_boosts_dramatic_weight() {
        let mut knowledge = HashMap::new();
        knowledge.insert(
            "alice".to_string(),
            vec![mock_knowledge("knowledge", "strict_secret")],
        );
        knowledge.insert("bob".to_string(), vec![]);

        // Mark the knowledge target as linked to a Strict enforcement fact
        let mut enforcement_levels = HashMap::new();
        enforcement_levels.insert(
            "knowledge:strict_secret".to_string(),
            EnforcementLevel::Strict,
        );

        let provider = MockIronyDataProviderWithContext {
            characters: vec![
                mock_character("alice", "Alice"),
                mock_character("bob", "Bob"),
            ],
            knowledge,
            scene_count: 10,
            perception_contexts: HashMap::new(),
            enforcement_levels,
        };
        let service = IronyService::with_provider(Arc::new(provider));

        let report = service.generate_report(None, 0).await.unwrap();
        assert_eq!(report.total_asymmetries, 1);

        let asym = &report.asymmetries[0];
        // dramatic_weight = 10 (scenes_since) + 3.0 (Strict) = 13.0
        assert!(
            (asym.dramatic_weight - 13.0).abs() < 0.01,
            "Strict enforcement should add 3.0 to weight. Got {}",
            asym.dramatic_weight
        );
    }

    #[tokio::test]
    async fn test_irony_report_sorting_by_tension_and_enforcement() {
        // Two asymmetries: one plain, one with tension+enforcement
        let mut knowledge = HashMap::new();
        knowledge.insert(
            "alice".to_string(),
            vec![
                mock_knowledge("knowledge", "plain_fact"),
                mock_knowledge("knowledge", "critical_fact"),
            ],
        );
        knowledge.insert("bob".to_string(), vec![]);

        let mut perception_contexts = HashMap::new();
        perception_contexts.insert(
            ("alice".to_string(), "bob".to_string()),
            PairContext {
                tension_level: Some(8),
                feelings: None,
                history_notes: None,
            },
        );

        let mut enforcement_levels = HashMap::new();
        enforcement_levels.insert(
            "knowledge:critical_fact".to_string(),
            EnforcementLevel::Strict,
        );

        let provider = MockIronyDataProviderWithContext {
            characters: vec![
                mock_character("alice", "Alice"),
                mock_character("bob", "Bob"),
            ],
            knowledge,
            scene_count: 10,
            perception_contexts,
            enforcement_levels,
        };
        let service = IronyService::with_provider(Arc::new(provider));

        let report = service.generate_report(None, 0).await.unwrap();
        assert_eq!(report.total_asymmetries, 2);

        // critical_fact should sort first: 10 + 3.0 (tension) + 3.0 (Strict) = 16.0
        // plain_fact: 10 + 3.0 (tension) = 13.0
        assert!(
            report.asymmetries[0].dramatic_weight > report.asymmetries[1].dramatic_weight,
            "Higher dramatic_weight ({}) should sort before lower ({})",
            report.asymmetries[0].dramatic_weight,
            report.asymmetries[1].dramatic_weight
        );
    }

    #[tokio::test]
    async fn test_narrative_opportunities_include_feelings_and_history() {
        let mut asym = make_asymmetry("Alice", "Bob", "the hidden truth", 10, "high");
        asym.feelings = Some("deep distrust".to_string());
        asym.history_notes = Some("Former friends turned enemies".to_string());

        let opportunities = generate_narrative_opportunities(&[asym], 5);
        assert_eq!(opportunities.len(), 1);
        assert!(
            opportunities[0].contains("[feels: deep distrust]"),
            "Should include feelings: {}",
            opportunities[0]
        );
        assert!(
            opportunities[0].contains("[history: Former friends turned enemies]"),
            "Should include history: {}",
            opportunities[0]
        );
    }

    #[tokio::test]
    async fn test_irony_service_threshold_filters() {
        let mut knowledge = HashMap::new();
        knowledge.insert(
            "alice".to_string(),
            vec![mock_knowledge("knowledge", "secret")],
        );
        knowledge.insert("bob".to_string(), vec![]);

        let provider = MockIronyDataProvider {
            characters: vec![
                mock_character("alice", "Alice"),
                mock_character("bob", "Bob"),
            ],
            knowledge,
            scene_count: 3, // scenes_since = 3
        };
        let service = IronyService::with_provider(Arc::new(provider));

        // Threshold of 5 should filter out (scenes_since = 3)
        let report = service.generate_report(None, 5).await.unwrap();
        assert_eq!(report.total_asymmetries, 0);

        // Threshold of 2 should include
        let report = service.generate_report(None, 2).await.unwrap();
        assert_eq!(report.total_asymmetries, 1);
    }

    // -- detect_asymmetries enrichment tests --

    #[tokio::test]
    async fn test_detect_asymmetries_enriched_with_tension() {
        let mut knowledge = HashMap::new();
        knowledge.insert(
            "alice".to_string(),
            vec![mock_knowledge("knowledge", "the_secret")],
        );
        knowledge.insert("bob".to_string(), vec![]);

        let mut perception_contexts = HashMap::new();
        perception_contexts.insert(
            ("alice".to_string(), "bob".to_string()),
            PairContext {
                tension_level: Some(8),
                feelings: Some("cold suspicion".to_string()),
                history_notes: Some("Former partners".to_string()),
            },
        );

        let provider = MockIronyDataProviderWithContext {
            characters: vec![
                mock_character("alice", "Alice"),
                mock_character("bob", "Bob"),
            ],
            knowledge,
            scene_count: 5,
            perception_contexts,
            enforcement_levels: HashMap::new(),
        };
        let service = IronyService::with_provider(Arc::new(provider));

        let asymmetries = service.detect_asymmetries("alice", "bob").await.unwrap();
        assert_eq!(asymmetries.len(), 1);

        let asym = &asymmetries[0];
        assert_eq!(asym.tension_level, Some(8));
        assert_eq!(asym.feelings.as_deref(), Some("cold suspicion"));
        assert_eq!(asym.history_notes.as_deref(), Some("Former partners"));
        // dramatic_weight = 5 (scenes_since) + 3.0 (tension >= 7) = 8.0
        assert!(
            (asym.dramatic_weight - 8.0).abs() < 0.01,
            "Expected 8.0, got {}",
            asym.dramatic_weight
        );
    }

    #[tokio::test]
    async fn test_detect_asymmetries_enforcement_boost() {
        let mut knowledge = HashMap::new();
        knowledge.insert(
            "alice".to_string(),
            vec![mock_knowledge("knowledge", "strict_rule")],
        );
        knowledge.insert("bob".to_string(), vec![]);

        let mut enforcement_levels = HashMap::new();
        enforcement_levels.insert(
            "knowledge:strict_rule".to_string(),
            EnforcementLevel::Strict,
        );

        let provider = MockIronyDataProviderWithContext {
            characters: vec![
                mock_character("alice", "Alice"),
                mock_character("bob", "Bob"),
            ],
            knowledge,
            scene_count: 5,
            perception_contexts: HashMap::new(),
            enforcement_levels,
        };
        let service = IronyService::with_provider(Arc::new(provider));

        let asymmetries = service.detect_asymmetries("alice", "bob").await.unwrap();
        assert_eq!(asymmetries.len(), 1);

        let asym = &asymmetries[0];
        // dramatic_weight = 5 (scenes_since) + 3.0 (Strict) = 8.0
        assert!(
            (asym.dramatic_weight - 8.0).abs() < 0.01,
            "Strict enforcement should boost weight to 8.0, got {}",
            asym.dramatic_weight
        );
    }
}
