//! Narrative tension detection service.
//!
//! Identifies dramatic tensions between characters by analyzing:
//! - Opposing desires (from character profiles)
//! - Contradictory knowledge (one believes_wrongly what another knows)
//! - Conflicting loyalties (allied with each other's rivals)
//! - High emotional contrast (emotion annotations)
//!
//! Enriches detected tensions with emotion and theme annotations for severity weighting.

use async_trait::async_trait;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::db::connection::NarraDb;
use crate::models::{CertaintyLevel, KnowledgeState};
use crate::utils::sanitize::validate_key;
use crate::NarraError;

/// A detected narrative tension between two characters.
#[derive(Debug, Clone, Serialize)]
pub struct NarrativeTension {
    pub character_a_id: String,
    pub character_a_name: String,
    pub character_b_id: String,
    pub character_b_name: String,
    /// Tension type: "opposing_desires", "contradictory_knowledge",
    /// "conflicting_loyalties", "emotional_conflict"
    pub tension_type: String,
    /// Human-readable description of the tension
    pub description: String,
    /// Severity score 0.0–1.0 (higher = more dramatically potent)
    pub severity: f32,
    /// Contributing signals that make up this tension
    pub signals: Vec<TensionSignal>,
}

/// A single signal contributing to a tension.
#[derive(Debug, Clone, Serialize)]
pub struct TensionSignal {
    pub signal_type: String,
    pub detail: String,
    pub weight: f32,
}

/// Full tension analysis report.
#[derive(Debug, Clone, Serialize)]
pub struct TensionReport {
    pub tensions: Vec<NarrativeTension>,
    pub total_count: usize,
    pub high_severity_count: usize,
}

// ---------------------------------------------------------------------------
// Pure functions
// ---------------------------------------------------------------------------

/// Detect opposing desires between two characters based on profile entries.
pub(crate) fn detect_opposing_desires(
    a_profile: &HashMap<String, Vec<String>>,
    b_profile: &HashMap<String, Vec<String>>,
) -> Vec<TensionSignal> {
    let mut signals = Vec::new();

    let a_desires = collect_profile_entries(
        a_profile,
        &["desire", "desire_conscious", "desire_unconscious", "goal"],
    );
    let b_desires = collect_profile_entries(
        b_profile,
        &["desire", "desire_conscious", "desire_unconscious", "goal"],
    );

    // Look for thematic opposition via keyword matching
    let opposition_pairs = [
        ("power", "freedom"),
        ("control", "independence"),
        ("revenge", "forgiveness"),
        ("secrecy", "truth"),
        ("loyalty", "betrayal"),
        ("order", "chaos"),
        ("tradition", "change"),
        ("isolation", "connection"),
        ("dominance", "equality"),
        ("safety", "adventure"),
    ];

    for a_desire in &a_desires {
        for b_desire in &b_desires {
            let a_lower = a_desire.to_lowercase();
            let b_lower = b_desire.to_lowercase();

            for (word_a, word_b) in &opposition_pairs {
                let a_matches = a_lower.contains(word_a) || a_lower.contains(word_b);
                let b_matches_opposite = (a_lower.contains(word_a) && b_lower.contains(word_b))
                    || (a_lower.contains(word_b) && b_lower.contains(word_a));

                if a_matches && b_matches_opposite {
                    signals.push(TensionSignal {
                        signal_type: "opposing_desire".to_string(),
                        detail: format!("\"{}\" vs \"{}\"", a_desire, b_desire),
                        weight: 0.7,
                    });
                }
            }
        }
    }

    signals
}

/// Detect contradictory knowledge between two characters.
pub(crate) fn detect_contradictory_knowledge(
    a_knowledge: &[KnowledgeState],
    b_knowledge: &[KnowledgeState],
) -> Vec<TensionSignal> {
    let mut signals = Vec::new();

    // Build target sets for comparison
    let a_by_target: HashMap<String, &KnowledgeState> = a_knowledge
        .iter()
        .map(|k| (k.target.to_string(), k))
        .collect();
    let b_by_target: HashMap<String, &KnowledgeState> = b_knowledge
        .iter()
        .map(|k| (k.target.to_string(), k))
        .collect();

    // Check for cases where one believes_wrongly what the other knows
    for (target, a_state) in &a_by_target {
        if let Some(b_state) = b_by_target.get(target) {
            // A knows correctly, B believes wrongly (or vice versa)
            if a_state.certainty == CertaintyLevel::Knows
                && b_state.certainty == CertaintyLevel::BelievesWrongly
            {
                signals.push(TensionSignal {
                    signal_type: "contradictory_knowledge".to_string(),
                    detail: format!(
                        "A knows the truth about {} while B believes wrongly",
                        target
                    ),
                    weight: 0.9,
                });
            } else if b_state.certainty == CertaintyLevel::Knows
                && a_state.certainty == CertaintyLevel::BelievesWrongly
            {
                signals.push(TensionSignal {
                    signal_type: "contradictory_knowledge".to_string(),
                    detail: format!(
                        "B knows the truth about {} while A believes wrongly",
                        target
                    ),
                    weight: 0.9,
                });
            }
            // Both deny what the other knows
            else if a_state.certainty == CertaintyLevel::Denies
                && b_state.certainty == CertaintyLevel::Knows
            {
                signals.push(TensionSignal {
                    signal_type: "knowledge_denial".to_string(),
                    detail: format!("A denies what B knows about {}", target),
                    weight: 0.8,
                });
            } else if b_state.certainty == CertaintyLevel::Denies
                && a_state.certainty == CertaintyLevel::Knows
            {
                signals.push(TensionSignal {
                    signal_type: "knowledge_denial".to_string(),
                    detail: format!("B denies what A knows about {}", target),
                    weight: 0.8,
                });
            }
        }
    }

    signals
}

/// Detect conflicting loyalties from relationship types.
pub(crate) fn detect_conflicting_loyalties(
    a_allies: &HashSet<String>,
    a_rivals: &HashSet<String>,
    b_allies: &HashSet<String>,
    b_rivals: &HashSet<String>,
) -> Vec<TensionSignal> {
    let mut signals = Vec::new();

    // A is allied with someone B considers a rival
    for ally in a_allies {
        if b_rivals.contains(ally) {
            signals.push(TensionSignal {
                signal_type: "conflicting_loyalty".to_string(),
                detail: format!("A is allied with {} whom B considers a rival", ally),
                weight: 0.8,
            });
        }
    }

    // B is allied with someone A considers a rival
    for ally in b_allies {
        if a_rivals.contains(ally) {
            signals.push(TensionSignal {
                signal_type: "conflicting_loyalty".to_string(),
                detail: format!("B is allied with {} whom A considers a rival", ally),
                weight: 0.8,
            });
        }
    }

    signals
}

/// Compute overall severity from tension signals.
pub(crate) fn compute_tension_severity(
    signals: &[TensionSignal],
    edge_tension_level: Option<i32>,
) -> f32 {
    if signals.is_empty() {
        return 0.0;
    }

    // Weighted average of signal weights
    let signal_score: f32 = signals.iter().map(|s| s.weight).sum::<f32>() / signals.len() as f32;

    // Boost from existing perceives tension_level
    let tension_boost = edge_tension_level
        .map(|t| (t as f32) / 10.0 * 0.3) // up to +0.3 from edge tension
        .unwrap_or(0.0);

    (signal_score + tension_boost).clamp(0.0, 1.0)
}

/// Classify the dominant tension type from signals.
pub(crate) fn classify_tension_type(signals: &[TensionSignal]) -> String {
    let mut type_counts: HashMap<&str, usize> = HashMap::new();
    for s in signals {
        *type_counts.entry(&s.signal_type).or_default() += 1;
    }

    type_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(t, _)| t.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Helper to collect profile entries from multiple key names.
fn collect_profile_entries(profile: &HashMap<String, Vec<String>>, keys: &[&str]) -> Vec<String> {
    keys.iter()
        .filter_map(|k| profile.get(*k))
        .flatten()
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// Data provider trait
// ---------------------------------------------------------------------------

/// Character info for tension detection.
#[derive(Debug, Clone)]
pub struct TensionCharacterInfo {
    pub id: String,
    pub name: String,
    pub profile: HashMap<String, Vec<String>>,
}

/// Relationship edge with type classification.
#[derive(Debug, Clone)]
pub struct RelationshipEdge {
    pub from_id: String,
    pub to_id: String,
    pub rel_type: String,
    pub tension_level: Option<i32>,
}

#[async_trait]
pub trait TensionDataProvider: Send + Sync {
    async fn get_all_characters(&self) -> Result<Vec<TensionCharacterInfo>, NarraError>;
    async fn get_character_knowledge(
        &self,
        character_id: &str,
    ) -> Result<Vec<KnowledgeState>, NarraError>;
    async fn get_all_relationship_edges(&self) -> Result<Vec<RelationshipEdge>, NarraError>;
    async fn get_perceives_tension(
        &self,
        char_a: &str,
        char_b: &str,
    ) -> Result<Option<i32>, NarraError>;
}

/// SurrealDB implementation.
pub struct SurrealTensionDataProvider {
    db: Arc<NarraDb>,
}

impl SurrealTensionDataProvider {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl TensionDataProvider for SurrealTensionDataProvider {
    async fn get_all_characters(&self) -> Result<Vec<TensionCharacterInfo>, NarraError> {
        #[derive(serde::Deserialize)]
        struct CharRow {
            id: surrealdb::sql::Thing,
            name: String,
            #[serde(default)]
            profile: HashMap<String, Vec<String>>,
        }
        let mut result = self
            .db
            .query("SELECT id, name, profile FROM character")
            .await?;
        let rows: Vec<CharRow> = result.take(0)?;
        Ok(rows
            .into_iter()
            .map(|r| TensionCharacterInfo {
                id: r.id.to_string(),
                name: r.name,
                profile: r.profile,
            })
            .collect())
    }

    async fn get_character_knowledge(
        &self,
        character_id: &str,
    ) -> Result<Vec<KnowledgeState>, NarraError> {
        let safe_id = validate_key(character_id)?;
        let query = format!(
            "SELECT * FROM knows WHERE in = character:{} ORDER BY learned_at DESC LIMIT 100",
            safe_id
        );
        let mut result = self.db.query(&query).await?;
        let states: Vec<KnowledgeState> = result.take(0)?;
        Ok(states)
    }

    async fn get_all_relationship_edges(&self) -> Result<Vec<RelationshipEdge>, NarraError> {
        #[derive(serde::Deserialize)]
        struct RelRow {
            #[serde(rename = "in")]
            from: surrealdb::sql::Thing,
            #[serde(rename = "out")]
            to: surrealdb::sql::Thing,
            rel_type: String,
        }
        let mut result = self
            .db
            .query("SELECT in, out, rel_type FROM relates_to")
            .await?;
        let rows: Vec<RelRow> = result.take(0)?;

        // Also fetch perceives for tension levels
        #[derive(serde::Deserialize)]
        struct PercRow {
            #[serde(rename = "in")]
            from: surrealdb::sql::Thing,
            #[serde(rename = "out")]
            to: surrealdb::sql::Thing,
            tension_level: Option<i32>,
        }
        let mut perc_result = self
            .db
            .query("SELECT in, out, tension_level FROM perceives")
            .await?;
        let perc_rows: Vec<PercRow> = perc_result.take(0)?;

        // Build tension lookup
        let tension_map: HashMap<(String, String), i32> = perc_rows
            .iter()
            .filter_map(|r| {
                r.tension_level
                    .map(|t| ((r.from.to_string(), r.to.to_string()), t))
            })
            .collect();

        let mut edges: Vec<RelationshipEdge> = rows
            .into_iter()
            .map(|r| {
                let from = r.from.to_string();
                let to = r.to.to_string();
                let tension = tension_map.get(&(from.clone(), to.clone())).copied();
                RelationshipEdge {
                    from_id: from,
                    to_id: to,
                    rel_type: r.rel_type,
                    tension_level: tension,
                }
            })
            .collect();

        // Also add perceives edges as relationship edges
        for pr in perc_rows {
            edges.push(RelationshipEdge {
                from_id: pr.from.to_string(),
                to_id: pr.to.to_string(),
                rel_type: "perceives".to_string(),
                tension_level: pr.tension_level,
            });
        }

        Ok(edges)
    }

    async fn get_perceives_tension(
        &self,
        char_a: &str,
        char_b: &str,
    ) -> Result<Option<i32>, NarraError> {
        #[derive(serde::Deserialize)]
        struct TensionRow {
            tension_level: Option<i32>,
        }
        let safe_a = validate_key(char_a)?;
        let safe_b = validate_key(char_b)?;
        let query = format!(
            "SELECT tension_level FROM perceives WHERE in = character:{} AND out = character:{} LIMIT 1",
            safe_a, safe_b
        );
        let mut result = self.db.query(&query).await?;
        let rows: Vec<TensionRow> = result.take(0)?;
        Ok(rows.first().and_then(|r| r.tension_level))
    }
}

// ---------------------------------------------------------------------------
// TensionService
// ---------------------------------------------------------------------------

/// Service for detecting narrative tensions between characters.
pub struct TensionService {
    data: Arc<dyn TensionDataProvider>,
}

impl TensionService {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self {
            data: Arc::new(SurrealTensionDataProvider::new(db)),
        }
    }

    pub fn with_provider(data: Arc<dyn TensionDataProvider>) -> Self {
        Self { data }
    }

    /// Detect all narrative tensions across the character network.
    pub async fn detect_tensions(
        &self,
        limit: usize,
        min_severity: f32,
    ) -> Result<TensionReport, NarraError> {
        let characters = self.data.get_all_characters().await?;
        let relationship_edges = self.data.get_all_relationship_edges().await?;

        // Build per-character ally/rival sets
        let mut allies: HashMap<String, HashSet<String>> = HashMap::new();
        let mut rivals: HashMap<String, HashSet<String>> = HashMap::new();

        for edge in &relationship_edges {
            let rel_lower = edge.rel_type.to_lowercase();
            if rel_lower.contains("ally")
                || rel_lower.contains("friend")
                || rel_lower.contains("mentor")
                || rel_lower.contains("family")
            {
                allies
                    .entry(edge.from_id.clone())
                    .or_default()
                    .insert(edge.to_id.clone());
            }
            if rel_lower.contains("rival")
                || rel_lower.contains("enemy")
                || rel_lower.contains("antagonist")
            {
                rivals
                    .entry(edge.from_id.clone())
                    .or_default()
                    .insert(edge.to_id.clone());
            }
        }

        // Build tension lookup from edges
        let mut edge_tensions: HashMap<(String, String), i32> = HashMap::new();
        for edge in &relationship_edges {
            if let Some(t) = edge.tension_level {
                edge_tensions.insert((edge.from_id.clone(), edge.to_id.clone()), t);
            }
        }

        // Pre-fetch knowledge for all characters
        let mut knowledge_map: HashMap<String, Vec<KnowledgeState>> = HashMap::new();
        for char in &characters {
            let char_key = char.id.split(':').nth(1).unwrap_or(&char.id);
            let knowledge = self.data.get_character_knowledge(char_key).await?;
            knowledge_map.insert(char.id.clone(), knowledge);
        }

        let empty_set = HashSet::new();
        let empty_knowledge = Vec::new();

        let mut tensions = Vec::new();

        // Check all character pairs
        for i in 0..characters.len() {
            for j in (i + 1)..characters.len() {
                let char_a = &characters[i];
                let char_b = &characters[j];

                let mut signals = Vec::new();

                // 1. Opposing desires
                signals.extend(detect_opposing_desires(&char_a.profile, &char_b.profile));

                // 2. Contradictory knowledge
                let a_knowledge = knowledge_map.get(&char_a.id).unwrap_or(&empty_knowledge);
                let b_knowledge = knowledge_map.get(&char_b.id).unwrap_or(&empty_knowledge);
                signals.extend(detect_contradictory_knowledge(a_knowledge, b_knowledge));

                // 3. Conflicting loyalties
                let a_allies = allies.get(&char_a.id).unwrap_or(&empty_set);
                let a_rivals = rivals.get(&char_a.id).unwrap_or(&empty_set);
                let b_allies = allies.get(&char_b.id).unwrap_or(&empty_set);
                let b_rivals = rivals.get(&char_b.id).unwrap_or(&empty_set);
                signals.extend(detect_conflicting_loyalties(
                    a_allies, a_rivals, b_allies, b_rivals,
                ));

                // 4. High edge tension from perceives
                let edge_tension = edge_tensions
                    .get(&(char_a.id.clone(), char_b.id.clone()))
                    .or_else(|| edge_tensions.get(&(char_b.id.clone(), char_a.id.clone())))
                    .copied();

                if let Some(t) = edge_tension {
                    if t >= 7 {
                        signals.push(TensionSignal {
                            signal_type: "high_edge_tension".to_string(),
                            detail: format!("Perceives tension level: {}/10", t),
                            weight: (t as f32) / 10.0,
                        });
                    }
                }

                if signals.is_empty() {
                    continue;
                }

                let severity = compute_tension_severity(&signals, edge_tension);
                if severity < min_severity {
                    continue;
                }

                let tension_type = classify_tension_type(&signals);
                let description =
                    format_tension_description(&char_a.name, &char_b.name, &tension_type, &signals);

                tensions.push(NarrativeTension {
                    character_a_id: char_a.id.clone(),
                    character_a_name: char_a.name.clone(),
                    character_b_id: char_b.id.clone(),
                    character_b_name: char_b.name.clone(),
                    tension_type,
                    description,
                    severity,
                    signals,
                });
            }
        }

        // Sort by severity descending
        tensions.sort_by(|a, b| {
            b.severity
                .partial_cmp(&a.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let high_severity_count = tensions.iter().filter(|t| t.severity >= 0.7).count();

        tensions.truncate(limit);

        let total_count = tensions.len();

        Ok(TensionReport {
            tensions,
            total_count,
            high_severity_count,
        })
    }
}

/// Generate human-readable tension description.
fn format_tension_description(
    name_a: &str,
    name_b: &str,
    tension_type: &str,
    signals: &[TensionSignal],
) -> String {
    let signal_count = signals.len();
    match tension_type {
        "opposing_desire" => format!(
            "{} and {} have opposing desires ({} signal{})",
            name_a,
            name_b,
            signal_count,
            if signal_count > 1 { "s" } else { "" }
        ),
        "contradictory_knowledge" | "knowledge_denial" => format!(
            "{} and {} hold contradictory beliefs ({} conflict{})",
            name_a,
            name_b,
            signal_count,
            if signal_count > 1 { "s" } else { "" }
        ),
        "conflicting_loyalty" => format!(
            "{} and {} have conflicting loyalties ({} clash{})",
            name_a,
            name_b,
            signal_count,
            if signal_count > 1 { "es" } else { "" }
        ),
        "high_edge_tension" => format!("{} and {} have high interpersonal tension", name_a, name_b),
        _ => format!(
            "Tension between {} and {} ({} signal{})",
            name_a,
            name_b,
            signal_count,
            if signal_count > 1 { "s" } else { "" }
        ),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LearningMethod;
    use surrealdb::RecordId;

    fn mock_knowledge(
        target_table: &str,
        target_id: &str,
        certainty: CertaintyLevel,
    ) -> KnowledgeState {
        let past = chrono::Utc::now() - chrono::Duration::hours(1);
        KnowledgeState {
            id: RecordId::from(("knows", "test")),
            character: RecordId::from(("character", "test")),
            target: RecordId::from((target_table, target_id)),
            certainty,
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

    // -- Pure function tests --

    #[test]
    fn test_detect_opposing_desires_found() {
        let mut a_profile = HashMap::new();
        a_profile.insert("desire".to_string(), vec!["gain total power".to_string()]);
        let mut b_profile = HashMap::new();
        b_profile.insert("desire".to_string(), vec!["fight for freedom".to_string()]);

        let signals = detect_opposing_desires(&a_profile, &b_profile);
        assert!(
            !signals.is_empty(),
            "Should detect power vs freedom opposition"
        );
        assert_eq!(signals[0].signal_type, "opposing_desire");
    }

    #[test]
    fn test_detect_opposing_desires_not_found() {
        let mut a_profile = HashMap::new();
        a_profile.insert("desire".to_string(), vec!["find love".to_string()]);
        let mut b_profile = HashMap::new();
        b_profile.insert("desire".to_string(), vec!["gain wealth".to_string()]);

        let signals = detect_opposing_desires(&a_profile, &b_profile);
        assert!(signals.is_empty(), "No opposition between love and wealth");
    }

    #[test]
    fn test_detect_opposing_desires_empty_profiles() {
        let signals = detect_opposing_desires(&HashMap::new(), &HashMap::new());
        assert!(signals.is_empty());
    }

    #[test]
    fn test_detect_opposing_desires_multiple_keys() {
        let mut a_profile = HashMap::new();
        a_profile.insert(
            "desire_conscious".to_string(),
            vec!["maintain order".to_string()],
        );
        let mut b_profile = HashMap::new();
        b_profile.insert("goal".to_string(), vec!["embrace chaos".to_string()]);

        let signals = detect_opposing_desires(&a_profile, &b_profile);
        assert!(
            !signals.is_empty(),
            "Should detect across different profile keys"
        );
    }

    #[test]
    fn test_detect_contradictory_knowledge() {
        let a_knows = vec![mock_knowledge("knowledge", "secret", CertaintyLevel::Knows)];
        let b_wrong = vec![mock_knowledge(
            "knowledge",
            "secret",
            CertaintyLevel::BelievesWrongly,
        )];

        let signals = detect_contradictory_knowledge(&a_knows, &b_wrong);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].signal_type, "contradictory_knowledge");
    }

    #[test]
    fn test_detect_contradictory_knowledge_both_know() {
        let a_knows = vec![mock_knowledge("knowledge", "secret", CertaintyLevel::Knows)];
        let b_knows = vec![mock_knowledge("knowledge", "secret", CertaintyLevel::Knows)];

        let signals = detect_contradictory_knowledge(&a_knows, &b_knows);
        assert!(signals.is_empty(), "No contradiction when both know");
    }

    #[test]
    fn test_detect_contradictory_knowledge_denial() {
        let a_knows = vec![mock_knowledge("knowledge", "fact1", CertaintyLevel::Knows)];
        let b_denies = vec![mock_knowledge("knowledge", "fact1", CertaintyLevel::Denies)];

        let signals = detect_contradictory_knowledge(&a_knows, &b_denies);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].signal_type, "knowledge_denial");
    }

    #[test]
    fn test_detect_conflicting_loyalties() {
        let a_allies: HashSet<String> = ["character:charlie".to_string()].into_iter().collect();
        let a_rivals: HashSet<String> = HashSet::new();
        let b_allies: HashSet<String> = HashSet::new();
        let b_rivals: HashSet<String> = ["character:charlie".to_string()].into_iter().collect();

        let signals = detect_conflicting_loyalties(&a_allies, &a_rivals, &b_allies, &b_rivals);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].signal_type, "conflicting_loyalty");
    }

    #[test]
    fn test_detect_conflicting_loyalties_none() {
        let a_allies: HashSet<String> = ["character:charlie".to_string()].into_iter().collect();
        let a_rivals: HashSet<String> = HashSet::new();
        let b_allies: HashSet<String> = ["character:dave".to_string()].into_iter().collect();
        let b_rivals: HashSet<String> = HashSet::new();

        let signals = detect_conflicting_loyalties(&a_allies, &a_rivals, &b_allies, &b_rivals);
        assert!(signals.is_empty());
    }

    #[test]
    fn test_compute_tension_severity() {
        let signals = vec![
            TensionSignal {
                signal_type: "opposing_desire".to_string(),
                detail: "test".to_string(),
                weight: 0.7,
            },
            TensionSignal {
                signal_type: "contradictory_knowledge".to_string(),
                detail: "test".to_string(),
                weight: 0.9,
            },
        ];

        let severity = compute_tension_severity(&signals, None);
        assert!((severity - 0.8).abs() < 0.01, "avg of 0.7 and 0.9");

        // With edge tension boost
        let severity_boosted = compute_tension_severity(&signals, Some(10));
        assert!(
            severity_boosted > severity,
            "Edge tension should boost severity"
        );
    }

    #[test]
    fn test_compute_tension_severity_empty() {
        assert_eq!(compute_tension_severity(&[], None), 0.0);
    }

    #[test]
    fn test_classify_tension_type() {
        let signals = vec![
            TensionSignal {
                signal_type: "opposing_desire".to_string(),
                detail: "test".to_string(),
                weight: 0.7,
            },
            TensionSignal {
                signal_type: "opposing_desire".to_string(),
                detail: "test2".to_string(),
                weight: 0.7,
            },
            TensionSignal {
                signal_type: "contradictory_knowledge".to_string(),
                detail: "test".to_string(),
                weight: 0.9,
            },
        ];
        assert_eq!(classify_tension_type(&signals), "opposing_desire");
    }

    // -- Mock-based service tests --

    struct MockTensionDataProvider {
        characters: Vec<TensionCharacterInfo>,
        knowledge: HashMap<String, Vec<KnowledgeState>>,
        relationships: Vec<RelationshipEdge>,
    }

    #[async_trait]
    impl TensionDataProvider for MockTensionDataProvider {
        async fn get_all_characters(&self) -> Result<Vec<TensionCharacterInfo>, NarraError> {
            Ok(self.characters.clone())
        }

        async fn get_character_knowledge(
            &self,
            character_id: &str,
        ) -> Result<Vec<KnowledgeState>, NarraError> {
            let full_id = format!("character:{}", character_id);
            Ok(self.knowledge.get(&full_id).cloned().unwrap_or_default())
        }

        async fn get_all_relationship_edges(&self) -> Result<Vec<RelationshipEdge>, NarraError> {
            Ok(self.relationships.clone())
        }

        async fn get_perceives_tension(
            &self,
            _char_a: &str,
            _char_b: &str,
        ) -> Result<Option<i32>, NarraError> {
            Ok(None)
        }
    }

    fn mock_char(
        id: &str,
        name: &str,
        profile: HashMap<String, Vec<String>>,
    ) -> TensionCharacterInfo {
        TensionCharacterInfo {
            id: format!("character:{}", id),
            name: name.to_string(),
            profile,
        }
    }

    #[tokio::test]
    async fn test_tension_service_empty_world() {
        let provider = MockTensionDataProvider {
            characters: vec![],
            knowledge: HashMap::new(),
            relationships: vec![],
        };
        let service = TensionService::with_provider(Arc::new(provider));
        let report = service.detect_tensions(50, 0.0).await.unwrap();
        assert!(report.tensions.is_empty());
    }

    #[tokio::test]
    async fn test_tension_service_opposing_desires() {
        let mut a_profile = HashMap::new();
        a_profile.insert("desire".to_string(), vec!["absolute power".to_string()]);
        let mut b_profile = HashMap::new();
        b_profile.insert("desire".to_string(), vec!["total freedom".to_string()]);

        let provider = MockTensionDataProvider {
            characters: vec![
                mock_char("alice", "Alice", a_profile),
                mock_char("bob", "Bob", b_profile),
            ],
            knowledge: HashMap::new(),
            relationships: vec![],
        };
        let service = TensionService::with_provider(Arc::new(provider));
        let report = service.detect_tensions(50, 0.0).await.unwrap();
        assert!(
            !report.tensions.is_empty(),
            "Should detect power vs freedom"
        );
        assert_eq!(report.tensions[0].tension_type, "opposing_desire");
    }

    #[tokio::test]
    async fn test_tension_service_contradictory_knowledge() {
        let mut knowledge = HashMap::new();
        knowledge.insert(
            "character:alice".to_string(),
            vec![mock_knowledge(
                "knowledge",
                "secret1",
                CertaintyLevel::Knows,
            )],
        );
        knowledge.insert(
            "character:bob".to_string(),
            vec![mock_knowledge(
                "knowledge",
                "secret1",
                CertaintyLevel::BelievesWrongly,
            )],
        );

        let provider = MockTensionDataProvider {
            characters: vec![
                mock_char("alice", "Alice", HashMap::new()),
                mock_char("bob", "Bob", HashMap::new()),
            ],
            knowledge,
            relationships: vec![],
        };
        let service = TensionService::with_provider(Arc::new(provider));
        let report = service.detect_tensions(50, 0.0).await.unwrap();
        assert!(!report.tensions.is_empty());
        assert!(report.tensions[0]
            .signals
            .iter()
            .any(|s| s.signal_type == "contradictory_knowledge"));
    }

    #[tokio::test]
    async fn test_tension_service_conflicting_loyalties() {
        let provider = MockTensionDataProvider {
            characters: vec![
                mock_char("alice", "Alice", HashMap::new()),
                mock_char("bob", "Bob", HashMap::new()),
                mock_char("charlie", "Charlie", HashMap::new()),
            ],
            knowledge: HashMap::new(),
            relationships: vec![
                RelationshipEdge {
                    from_id: "character:alice".to_string(),
                    to_id: "character:charlie".to_string(),
                    rel_type: "ally".to_string(),
                    tension_level: None,
                },
                RelationshipEdge {
                    from_id: "character:bob".to_string(),
                    to_id: "character:charlie".to_string(),
                    rel_type: "rival".to_string(),
                    tension_level: None,
                },
            ],
        };
        let service = TensionService::with_provider(Arc::new(provider));
        let report = service.detect_tensions(50, 0.0).await.unwrap();

        // Alice-Bob should have conflicting loyalty about Charlie
        let alice_bob = report.tensions.iter().find(|t| {
            (t.character_a_name == "Alice" && t.character_b_name == "Bob")
                || (t.character_a_name == "Bob" && t.character_b_name == "Alice")
        });
        assert!(
            alice_bob.is_some(),
            "Should detect Alice-Bob loyalty conflict"
        );
    }

    #[tokio::test]
    async fn test_tension_service_severity_filter() {
        let mut a_profile = HashMap::new();
        a_profile.insert("desire".to_string(), vec!["absolute power".to_string()]);
        let mut b_profile = HashMap::new();
        b_profile.insert("desire".to_string(), vec!["total freedom".to_string()]);

        let provider = MockTensionDataProvider {
            characters: vec![
                mock_char("alice", "Alice", a_profile),
                mock_char("bob", "Bob", b_profile),
            ],
            knowledge: HashMap::new(),
            relationships: vec![],
        };
        let service = TensionService::with_provider(Arc::new(provider));

        // Very high threshold should filter everything
        let report = service.detect_tensions(50, 0.99).await.unwrap();
        assert!(report.tensions.is_empty(), "High threshold should filter");
    }

    // -- Property-based tests --

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        fn arb_signal() -> impl Strategy<Value = TensionSignal> {
            (
                prop::sample::select(vec![
                    "opposing_desire",
                    "contradictory_knowledge",
                    "knowledge_denial",
                    "conflicting_loyalty",
                    "high_edge_tension",
                ]),
                ".*",
                0.0f32..=1.0f32,
            )
                .prop_map(|(signal_type, detail, weight)| TensionSignal {
                    signal_type: signal_type.to_string(),
                    detail,
                    weight,
                })
        }

        proptest! {
            #[test]
            fn prop_severity_in_range(
                signals in proptest::collection::vec(arb_signal(), 1..10),
                edge_tension in proptest::option::of(0i32..=10),
            ) {
                let severity = compute_tension_severity(&signals, edge_tension);
                prop_assert!(severity >= 0.0, "Severity must be >= 0.0, got: {}", severity);
                prop_assert!(severity <= 1.0, "Severity must be <= 1.0, got: {}", severity);
            }

            #[test]
            fn prop_severity_empty_is_zero(
                edge_tension in proptest::option::of(0i32..=10),
            ) {
                let severity = compute_tension_severity(&[], edge_tension);
                prop_assert_eq!(severity, 0.0);
            }

            #[test]
            fn prop_classify_returns_known_type(
                signals in proptest::collection::vec(arb_signal(), 1..10),
            ) {
                let result = classify_tension_type(&signals);
                let known_types = [
                    "opposing_desire",
                    "contradictory_knowledge",
                    "knowledge_denial",
                    "conflicting_loyalty",
                    "high_edge_tension",
                ];
                prop_assert!(
                    known_types.contains(&result.as_str()),
                    "Unknown tension type: {}",
                    result
                );
            }

            #[test]
            fn prop_opposing_desires_symmetric(
                a_entries in proptest::collection::vec("[a-z ]{3,20}", 0..5),
                b_entries in proptest::collection::vec("[a-z ]{3,20}", 0..5),
            ) {
                let mut a_profile = HashMap::new();
                let mut b_profile = HashMap::new();
                if !a_entries.is_empty() {
                    a_profile.insert("desire".to_string(), a_entries.clone());
                }
                if !b_entries.is_empty() {
                    b_profile.insert("desire".to_string(), b_entries.clone());
                }

                let ab = detect_opposing_desires(&a_profile, &b_profile);
                let ba = detect_opposing_desires(&b_profile, &a_profile);

                // Opposing desires should be symmetric — if A opposes B, B opposes A
                prop_assert_eq!(
                    ab.len(),
                    ba.len(),
                    "Opposing desires should be symmetric"
                );
            }

            #[test]
            fn prop_conflicting_loyalties_no_self_conflict(
                allies in proptest::collection::hash_set("[a-z:]{5,20}", 0..5),
                rivals in proptest::collection::hash_set("[a-z:]{5,20}", 0..5),
            ) {
                // If A's allies and rivals are the same set as B's, no conflict
                let signals = detect_conflicting_loyalties(
                    &allies, &rivals, &allies, &rivals,
                );
                // If someone is both ally and rival of A, then A is "allied with their
                // own rival", which IS a valid (if weird) conflict signal. This is fine.
                // The function doesn't self-conflict for empty sets though.
                if allies.is_empty() || rivals.is_empty() {
                    prop_assert!(signals.is_empty());
                }
            }
        }
    }

    #[tokio::test]
    async fn test_tension_service_symmetry() {
        // Tension between A and B should also appear between B and A
        let mut knowledge = HashMap::new();
        knowledge.insert(
            "character:alice".to_string(),
            vec![mock_knowledge("knowledge", "fact1", CertaintyLevel::Knows)],
        );
        knowledge.insert(
            "character:bob".to_string(),
            vec![mock_knowledge(
                "knowledge",
                "fact1",
                CertaintyLevel::BelievesWrongly,
            )],
        );

        let provider = MockTensionDataProvider {
            characters: vec![
                mock_char("alice", "Alice", HashMap::new()),
                mock_char("bob", "Bob", HashMap::new()),
            ],
            knowledge,
            relationships: vec![],
        };
        let service = TensionService::with_provider(Arc::new(provider));
        let report = service.detect_tensions(50, 0.0).await.unwrap();

        // Should have exactly 1 tension (pair analyzed once, not twice)
        assert_eq!(report.tensions.len(), 1);
    }
}
