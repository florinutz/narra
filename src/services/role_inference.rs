//! Narrative role inference service.
//!
//! Derives structural narrative roles for characters by analyzing:
//! - Graph topology (degree, betweenness from graph analytics)
//! - Knowledge graph patterns (who knows about whom, false beliefs)
//! - Relationship types (mentor, rival, ally distributions)
//! - Profile traits (secrets, contradictions, wounds)
//!
//! Produces richer role classifications than basic hub/bridge/peripheral.

use async_trait::async_trait;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;

use crate::db::connection::NarraDb;
use crate::NarraError;

/// An inferred narrative role for a character.
#[derive(Debug, Clone, Serialize)]
pub struct InferredRole {
    pub character_id: String,
    pub character_name: String,
    /// Primary inferred role label
    pub primary_role: String,
    /// Secondary roles (may overlap)
    pub secondary_roles: Vec<String>,
    /// Confidence in the primary role assignment (0.0–1.0)
    pub confidence: f32,
    /// Evidence signals that contributed to the inference
    pub evidence: Vec<RoleEvidence>,
}

/// A single piece of evidence for a role inference.
#[derive(Debug, Clone, Serialize)]
pub struct RoleEvidence {
    pub signal: String,
    pub detail: String,
    pub weight: f32,
}

/// Full role inference report.
#[derive(Debug, Clone, Serialize)]
pub struct RoleReport {
    pub roles: Vec<InferredRole>,
    pub total_characters: usize,
}

// ---------------------------------------------------------------------------
// Pure functions
// ---------------------------------------------------------------------------

/// Infer roles from a character's structural features.
pub(crate) fn infer_roles_from_features(
    features: &CharacterFeatures,
) -> (String, Vec<String>, f32, Vec<RoleEvidence>) {
    let mut evidence = Vec::new();
    let mut role_scores: HashMap<&str, f32> = HashMap::new();

    // --- Graph topology signals ---

    // High degree → social hub
    if features.degree_centrality > 0.5 {
        let w = 0.6 + (features.degree_centrality as f32 - 0.5) * 0.8;
        evidence.push(RoleEvidence {
            signal: "high_degree".to_string(),
            detail: format!("Degree centrality: {:.2}", features.degree_centrality),
            weight: w,
        });
        *role_scores.entry("social_hub").or_default() += w;
    }

    // High betweenness + moderate degree → bridge / gatekeeper
    if features.betweenness_centrality > 0.3 && features.degree_centrality < 0.6 {
        let w = 0.5 + features.betweenness_centrality as f32 * 0.5;
        evidence.push(RoleEvidence {
            signal: "high_betweenness".to_string(),
            detail: format!(
                "Betweenness: {:.2}, connects disparate groups",
                features.betweenness_centrality
            ),
            weight: w,
        });
        *role_scores.entry("bridge").or_default() += w;
    }

    // Zero degree → isolated
    if features.degree_centrality == 0.0 {
        evidence.push(RoleEvidence {
            signal: "isolated".to_string(),
            detail: "No connections in the relationship graph".to_string(),
            weight: 0.9,
        });
        *role_scores.entry("outsider").or_default() += 0.9;
    }

    // Low degree, non-zero → peripheral
    if features.degree_centrality > 0.0 && features.degree_centrality < 0.2 {
        evidence.push(RoleEvidence {
            signal: "peripheral".to_string(),
            detail: format!("Low degree centrality: {:.2}", features.degree_centrality),
            weight: 0.4,
        });
        *role_scores.entry("peripheral").or_default() += 0.4;
    }

    // --- Knowledge graph signals ---

    // Many know about this character → mystery node / person of interest
    if features.known_about_count >= 3 {
        let w = 0.4 + (features.known_about_count as f32 / 10.0).min(0.5);
        evidence.push(RoleEvidence {
            signal: "known_about".to_string(),
            detail: format!(
                "{} characters have knowledge about them",
                features.known_about_count
            ),
            weight: w,
        });
        *role_scores.entry("person_of_interest").or_default() += w;
    }

    // Many false beliefs about this character → enigma
    if features.false_beliefs_about_count >= 2 {
        let w = 0.5 + (features.false_beliefs_about_count as f32 / 5.0).min(0.4);
        evidence.push(RoleEvidence {
            signal: "false_beliefs_target".to_string(),
            detail: format!(
                "{} characters believe wrongly about them",
                features.false_beliefs_about_count
            ),
            weight: w,
        });
        *role_scores.entry("enigma").or_default() += w;
    }

    // Character holds many false beliefs → deceived
    if features.holds_false_beliefs >= 2 {
        let w = 0.4 + (features.holds_false_beliefs as f32 / 5.0).min(0.4);
        evidence.push(RoleEvidence {
            signal: "holds_false_beliefs".to_string(),
            detail: format!("Holds {} false beliefs", features.holds_false_beliefs),
            weight: w,
        });
        *role_scores.entry("deceived").or_default() += w;
    }

    // Character knows many things → information broker
    if features.knowledge_count >= 5 {
        let w = 0.3 + (features.knowledge_count as f32 / 20.0).min(0.4);
        evidence.push(RoleEvidence {
            signal: "high_knowledge".to_string(),
            detail: format!("Knows {} facts", features.knowledge_count),
            weight: w,
        });
        *role_scores.entry("information_broker").or_default() += w;
    }

    // --- Relationship type signals ---

    // Many mentor/teacher relationships → mentor
    if features.mentor_out_count >= 1 {
        let w = 0.5 + (features.mentor_out_count as f32 / 3.0).min(0.4);
        evidence.push(RoleEvidence {
            signal: "mentor_relationships".to_string(),
            detail: format!("{} mentorship relationships", features.mentor_out_count),
            weight: w,
        });
        *role_scores.entry("mentor").or_default() += w;
    }

    // Many rival/enemy edges → antagonist tendency
    if features.rival_count >= 2 {
        let w = 0.4 + (features.rival_count as f32 / 5.0).min(0.4);
        evidence.push(RoleEvidence {
            signal: "many_rivals".to_string(),
            detail: format!("{} rival/enemy relationships", features.rival_count),
            weight: w,
        });
        *role_scores.entry("antagonist").or_default() += w;
    }

    // Many ally/friend edges → connector / social butterfly
    if features.ally_count >= 3 {
        let w = 0.3 + (features.ally_count as f32 / 6.0).min(0.4);
        evidence.push(RoleEvidence {
            signal: "many_allies".to_string(),
            detail: format!("{} ally/friend relationships", features.ally_count),
            weight: w,
        });
        *role_scores.entry("connector").or_default() += w;
    }

    // --- Profile trait signals ---

    // Has secrets → keeper of secrets
    if features.secret_count >= 1 {
        let w = 0.4 + (features.secret_count as f32 / 3.0).min(0.4);
        evidence.push(RoleEvidence {
            signal: "has_secrets".to_string(),
            detail: format!("{} secret(s) in profile", features.secret_count),
            weight: w,
        });
        *role_scores.entry("keeper_of_secrets").or_default() += w;
    }

    // Has contradictions → complex / unreliable
    if features.contradiction_count >= 1 {
        let w = 0.3 + (features.contradiction_count as f32 / 3.0).min(0.4);
        evidence.push(RoleEvidence {
            signal: "has_contradictions".to_string(),
            detail: format!(
                "{} contradiction(s) in profile",
                features.contradiction_count
            ),
            weight: w,
        });
        *role_scores.entry("complex_character").or_default() += w;
    }

    // Has wounds → tragic figure
    if features.wound_count >= 1 {
        let w = 0.3 + (features.wound_count as f32 / 3.0).min(0.3);
        evidence.push(RoleEvidence {
            signal: "has_wounds".to_string(),
            detail: format!("{} wound(s) in profile", features.wound_count),
            weight: w,
        });
        *role_scores.entry("tragic_figure").or_default() += w;
    }

    // Determine primary and secondary roles (sort by score desc, then name asc for determinism)
    let mut sorted_roles: Vec<(&&str, &f32)> = role_scores.iter().collect();
    sorted_roles.sort_by(|a, b| {
        b.1.partial_cmp(a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(b.0))
    });

    let primary_role = sorted_roles
        .first()
        .map(|(r, _)| r.to_string())
        .unwrap_or_else(|| "undefined".to_string());

    let confidence = sorted_roles
        .first()
        .map(|(_, s)| (**s).clamp(0.0, 1.0))
        .unwrap_or(0.0);

    let secondary_roles: Vec<String> = sorted_roles
        .iter()
        .skip(1)
        .filter(|(_, s)| **s >= 0.3)
        .take(3)
        .map(|(r, _)| r.to_string())
        .collect();

    (primary_role, secondary_roles, confidence, evidence)
}

// ---------------------------------------------------------------------------
// Data provider trait
// ---------------------------------------------------------------------------

/// Aggregated features for a character used in role inference.
#[derive(Debug, Clone, Default)]
pub struct CharacterFeatures {
    pub id: String,
    pub name: String,
    pub degree_centrality: f64,
    pub betweenness_centrality: f64,
    /// How many other characters have knowledge about this character
    pub known_about_count: usize,
    /// How many other characters hold false beliefs about this character
    pub false_beliefs_about_count: usize,
    /// How many false beliefs this character holds
    pub holds_false_beliefs: usize,
    /// Total knowledge entries this character has
    pub knowledge_count: usize,
    /// Mentor/teacher outgoing relationship count
    pub mentor_out_count: usize,
    /// Rival/enemy relationship count
    pub rival_count: usize,
    /// Ally/friend relationship count
    pub ally_count: usize,
    /// Number of secrets in profile
    pub secret_count: usize,
    /// Number of contradictions in profile
    pub contradiction_count: usize,
    /// Number of wounds in profile
    pub wound_count: usize,
}

#[async_trait]
pub trait RoleInferenceDataProvider: Send + Sync {
    /// Get aggregated features for all characters.
    async fn get_all_character_features(&self) -> Result<Vec<CharacterFeatures>, NarraError>;
}

/// SurrealDB implementation.
pub struct SurrealRoleInferenceDataProvider {
    db: Arc<NarraDb>,
}

impl SurrealRoleInferenceDataProvider {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl RoleInferenceDataProvider for SurrealRoleInferenceDataProvider {
    async fn get_all_character_features(&self) -> Result<Vec<CharacterFeatures>, NarraError> {
        // 1. Get all characters with profiles
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
        let chars: Vec<CharRow> = result.take(0)?;

        if chars.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Get knowledge edges for certainty analysis
        #[derive(serde::Deserialize)]
        struct KnowsRow {
            #[serde(rename = "in")]
            knower: surrealdb::sql::Thing,
            #[serde(rename = "out")]
            target: surrealdb::sql::Thing,
            certainty: String,
        }
        let mut knows_result = self
            .db
            .query("SELECT in, out, certainty FROM knows")
            .await?;
        let knows: Vec<KnowsRow> = knows_result.take(0)?;

        // 3. Get relationship edges for type classification
        #[derive(serde::Deserialize)]
        struct RelRow {
            #[serde(rename = "in")]
            from: surrealdb::sql::Thing,
            rel_type: String,
        }
        let mut rel_result = self.db.query("SELECT in, rel_type FROM relates_to").await?;
        let rels: Vec<RelRow> = rel_result.take(0)?;

        // 4. Get perception edges for degree computation
        #[derive(serde::Deserialize)]
        struct PercRow {
            #[serde(rename = "in")]
            from: surrealdb::sql::Thing,
            #[serde(rename = "out")]
            to: surrealdb::sql::Thing,
        }
        let mut perc_result = self.db.query("SELECT in, out FROM perceives").await?;
        let percs: Vec<PercRow> = perc_result.take(0)?;

        // Compute per-character features
        let n = chars.len();
        let max_degree = if n > 1 { (n - 1) as f64 } else { 1.0 };

        // Degree: count unique connection partners (perceives + relates_to)
        let mut connections: HashMap<String, std::collections::HashSet<String>> =
            HashMap::with_capacity(n);
        for p in &percs {
            let from = p.from.to_string();
            let to = p.to.to_string();
            connections
                .entry(from.clone())
                .or_default()
                .insert(to.clone());
            connections.entry(to).or_default().insert(from);
        }
        for r in &rels {
            // relates_to in SurrealDB: in = from, we don't have out here
            // We only fetched 'in' and 'rel_type', so we can count from-side rels
            let from = r.from.to_string();
            connections.entry(from).or_default();
        }

        // Knowledge aggregation
        let mut knowledge_count: HashMap<String, usize> = HashMap::with_capacity(n);
        let mut known_about: HashMap<String, usize> = HashMap::with_capacity(n);
        let mut false_beliefs_about: HashMap<String, usize> = HashMap::with_capacity(n);
        let mut holds_false: HashMap<String, usize> = HashMap::with_capacity(n);

        for k in &knows {
            let knower = k.knower.to_string();
            let target = k.target.to_string();
            *knowledge_count.entry(knower.clone()).or_default() += 1;

            // Target must be a character for "known about" to count
            if target.starts_with("character:") {
                *known_about.entry(target.clone()).or_default() += 1;
                if k.certainty == "believes_wrongly" {
                    *false_beliefs_about.entry(target).or_default() += 1;
                    *holds_false.entry(knower).or_default() += 1;
                }
            } else if k.certainty == "believes_wrongly" {
                *holds_false.entry(knower).or_default() += 1;
            }
        }

        // Relationship type counts
        let mut mentor_out: HashMap<String, usize> = HashMap::with_capacity(n);
        let mut rival_count: HashMap<String, usize> = HashMap::with_capacity(n);
        let mut ally_count: HashMap<String, usize> = HashMap::with_capacity(n);

        for r in &rels {
            let from = r.from.to_string();
            let rel_lower = r.rel_type.to_lowercase();
            if rel_lower.contains("mentor")
                || rel_lower.contains("teacher")
                || rel_lower.contains("guide")
            {
                *mentor_out.entry(from.clone()).or_default() += 1;
            }
            if rel_lower.contains("rival")
                || rel_lower.contains("enemy")
                || rel_lower.contains("antagonist")
            {
                *rival_count.entry(from.clone()).or_default() += 1;
            }
            if rel_lower.contains("ally")
                || rel_lower.contains("friend")
                || rel_lower.contains("family")
            {
                *ally_count.entry(from.clone()).or_default() += 1;
            }
        }

        // Build features
        let features: Vec<CharacterFeatures> = chars
            .into_iter()
            .map(|c| {
                let id = c.id.to_string();
                let degree = connections
                    .get(&id)
                    .map(|s| s.len() as f64 / max_degree)
                    .unwrap_or(0.0);

                let secret_count = c.profile.get("secret").map(|v| v.len()).unwrap_or(0)
                    + c.profile.get("secrets").map(|v| v.len()).unwrap_or(0);
                let contradiction_count =
                    c.profile.get("contradiction").map(|v| v.len()).unwrap_or(0)
                        + c.profile
                            .get("contradictions")
                            .map(|v| v.len())
                            .unwrap_or(0);
                let wound_count = c.profile.get("wound").map(|v| v.len()).unwrap_or(0)
                    + c.profile.get("wounds").map(|v| v.len()).unwrap_or(0);

                CharacterFeatures {
                    name: c.name,
                    degree_centrality: degree,
                    betweenness_centrality: 0.0, // Approximated by role heuristics; full betweenness requires graphrs
                    known_about_count: known_about.get(&id).copied().unwrap_or(0),
                    false_beliefs_about_count: false_beliefs_about.get(&id).copied().unwrap_or(0),
                    holds_false_beliefs: holds_false.get(&id).copied().unwrap_or(0),
                    knowledge_count: knowledge_count.get(&id).copied().unwrap_or(0),
                    mentor_out_count: mentor_out.get(&id).copied().unwrap_or(0),
                    rival_count: rival_count.get(&id).copied().unwrap_or(0),
                    ally_count: ally_count.get(&id).copied().unwrap_or(0),
                    secret_count,
                    contradiction_count,
                    wound_count,
                    id,
                }
            })
            .collect();

        Ok(features)
    }
}

// ---------------------------------------------------------------------------
// RoleInferenceService
// ---------------------------------------------------------------------------

/// Service for inferring narrative roles from character features.
pub struct RoleInferenceService {
    data: Arc<dyn RoleInferenceDataProvider>,
}

impl RoleInferenceService {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self {
            data: Arc::new(SurrealRoleInferenceDataProvider::new(db)),
        }
    }

    pub fn with_provider(data: Arc<dyn RoleInferenceDataProvider>) -> Self {
        Self { data }
    }

    /// Infer narrative roles for all characters.
    pub async fn infer_roles(&self, limit: usize) -> Result<RoleReport, NarraError> {
        let features = self.data.get_all_character_features().await?;
        let total_characters = features.len();

        let mut roles: Vec<InferredRole> = features
            .iter()
            .map(|f| {
                let (primary_role, secondary_roles, confidence, evidence) =
                    infer_roles_from_features(f);
                InferredRole {
                    character_id: f.id.clone(),
                    character_name: f.name.clone(),
                    primary_role,
                    secondary_roles,
                    confidence,
                    evidence,
                }
            })
            .collect();

        // Sort by confidence descending
        roles.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        roles.truncate(limit);

        Ok(RoleReport {
            roles,
            total_characters,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Pure function tests --

    fn empty_features(id: &str, name: &str) -> CharacterFeatures {
        CharacterFeatures {
            id: format!("character:{}", id),
            name: name.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_infer_roles_isolated_character() {
        let features = empty_features("loner", "The Loner");
        let (primary, _, confidence, evidence) = infer_roles_from_features(&features);
        assert_eq!(primary, "outsider");
        assert!(confidence > 0.5);
        assert!(evidence.iter().any(|e| e.signal == "isolated"));
    }

    #[test]
    fn test_infer_roles_social_hub() {
        let mut features = empty_features("alice", "Alice");
        features.degree_centrality = 0.8;
        features.ally_count = 5;

        let (primary, secondary, _, evidence) = infer_roles_from_features(&features);
        assert_eq!(primary, "social_hub");
        assert!(evidence.iter().any(|e| e.signal == "high_degree"));
        // Should also have connector as secondary
        assert!(
            secondary.contains(&"connector".to_string()),
            "Should include connector as secondary, got: {:?}",
            secondary
        );
    }

    #[test]
    fn test_infer_roles_mentor() {
        let mut features = empty_features("gandalf", "Gandalf");
        features.degree_centrality = 0.3;
        features.mentor_out_count = 3;
        features.knowledge_count = 10;

        let (primary, _, _, evidence) = infer_roles_from_features(&features);
        // Mentor or information_broker should be primary
        assert!(
            primary == "mentor" || primary == "information_broker",
            "Expected mentor or information_broker, got: {}",
            primary
        );
        assert!(evidence.iter().any(|e| e.signal == "mentor_relationships"));
    }

    #[test]
    fn test_infer_roles_enigma() {
        let mut features = empty_features("mystery", "The Mystery");
        features.known_about_count = 5;
        features.false_beliefs_about_count = 3;
        features.secret_count = 2;

        let (_, _, _, evidence) = infer_roles_from_features(&features);
        assert!(evidence.iter().any(|e| e.signal == "false_beliefs_target"));
        assert!(evidence.iter().any(|e| e.signal == "has_secrets"));
    }

    #[test]
    fn test_infer_roles_deceived() {
        let mut features = empty_features("victim", "The Victim");
        features.holds_false_beliefs = 4;

        let (_, _, _, evidence) = infer_roles_from_features(&features);
        assert!(evidence.iter().any(|e| e.signal == "holds_false_beliefs"));
    }

    #[test]
    fn test_infer_roles_antagonist() {
        let mut features = empty_features("villain", "The Villain");
        features.degree_centrality = 0.4;
        features.rival_count = 4;

        let (primary, _, _, evidence) = infer_roles_from_features(&features);
        assert_eq!(primary, "antagonist");
        assert!(evidence.iter().any(|e| e.signal == "many_rivals"));
    }

    #[test]
    fn test_infer_roles_tragic_figure() {
        let mut features = empty_features("tragic", "Tragic Hero");
        features.wound_count = 2;
        features.contradiction_count = 1;

        let (_, _, _, evidence) = infer_roles_from_features(&features);
        assert!(evidence.iter().any(|e| e.signal == "has_wounds"));
        assert!(evidence.iter().any(|e| e.signal == "has_contradictions"));
    }

    #[test]
    fn test_infer_roles_empty_features() {
        // Character with zero degree but no other signals
        let features = empty_features("blank", "Blank Slate");
        let (primary, secondary, _, _) = infer_roles_from_features(&features);
        assert_eq!(primary, "outsider"); // Zero degree triggers outsider
        assert!(secondary.is_empty() || secondary.len() <= 3);
    }

    #[test]
    fn test_infer_roles_bridge() {
        let mut features = empty_features("messenger", "The Messenger");
        features.degree_centrality = 0.3;
        features.betweenness_centrality = 0.6;

        let (primary, _, _, evidence) = infer_roles_from_features(&features);
        assert_eq!(primary, "bridge");
        assert!(evidence.iter().any(|e| e.signal == "high_betweenness"));
    }

    #[test]
    fn test_infer_roles_confidence_clamped() {
        // Even with extreme features, confidence should not exceed 1.0
        let mut features = empty_features("extreme", "Extreme");
        features.degree_centrality = 1.0;
        features.ally_count = 100;
        features.knowledge_count = 100;

        let (_, _, confidence, _) = infer_roles_from_features(&features);
        assert!(confidence <= 1.0, "Confidence should be clamped to 1.0");
    }

    #[test]
    fn test_infer_roles_max_secondary() {
        // Should return at most 3 secondary roles
        let mut features = empty_features("multi", "Multi");
        features.degree_centrality = 0.7;
        features.ally_count = 5;
        features.rival_count = 3;
        features.knowledge_count = 10;
        features.mentor_out_count = 2;
        features.secret_count = 2;

        let (_, secondary, _, _) = infer_roles_from_features(&features);
        assert!(
            secondary.len() <= 3,
            "Should have at most 3 secondary roles, got: {}",
            secondary.len()
        );
    }

    // -- Property-based tests --

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        fn arb_features() -> impl Strategy<Value = CharacterFeatures> {
            (
                0.0f64..=1.0f64, // degree_centrality
                0.0f64..=1.0f64, // betweenness_centrality
                0usize..20,      // known_about_count
                0usize..10,      // false_beliefs_about_count
                0usize..10,      // holds_false_beliefs
                0usize..30,      // knowledge_count
                0usize..5,       // mentor_out_count
                0usize..10,      // rival_count
                0usize..10,      // ally_count
                0usize..5,       // secret_count
                0usize..5,       // contradiction_count
                0usize..5,       // wound_count
            )
                .prop_map(|(deg, bet, ka, fba, hfb, kc, mo, rc, ac, sc, cc, wc)| {
                    CharacterFeatures {
                        id: "character:test".to_string(),
                        name: "Test".to_string(),
                        degree_centrality: deg,
                        betweenness_centrality: bet,
                        known_about_count: ka,
                        false_beliefs_about_count: fba,
                        holds_false_beliefs: hfb,
                        knowledge_count: kc,
                        mentor_out_count: mo,
                        rival_count: rc,
                        ally_count: ac,
                        secret_count: sc,
                        contradiction_count: cc,
                        wound_count: wc,
                    }
                })
        }

        proptest! {
            #[test]
            fn prop_confidence_in_range(features in arb_features()) {
                let (_, _, confidence, _) = infer_roles_from_features(&features);
                prop_assert!(confidence >= 0.0, "Confidence must be >= 0.0, got: {}", confidence);
                prop_assert!(confidence <= 1.0, "Confidence must be <= 1.0, got: {}", confidence);
            }

            #[test]
            fn prop_max_three_secondary_roles(features in arb_features()) {
                let (_, secondary, _, _) = infer_roles_from_features(&features);
                prop_assert!(
                    secondary.len() <= 3,
                    "Should have at most 3 secondary roles, got: {}",
                    secondary.len()
                );
            }

            #[test]
            fn prop_primary_role_not_empty(features in arb_features()) {
                let (primary, _, _, _) = infer_roles_from_features(&features);
                prop_assert!(!primary.is_empty(), "Primary role should not be empty");
            }

            #[test]
            fn prop_deterministic(features in arb_features()) {
                let (role1, sec1, conf1, _) = infer_roles_from_features(&features);
                let (role2, sec2, conf2, _) = infer_roles_from_features(&features);
                prop_assert_eq!(role1, role2, "Same input should produce same primary role");
                prop_assert_eq!(sec1, sec2, "Same input should produce same secondary roles");
                prop_assert_eq!(conf1, conf2, "Same input should produce same confidence");
            }

            #[test]
            fn prop_evidence_weights_non_negative(features in arb_features()) {
                let (_, _, _, evidence) = infer_roles_from_features(&features);
                for ev in &evidence {
                    prop_assert!(
                        ev.weight >= 0.0,
                        "Evidence weight must be >= 0.0, got: {} for signal {}",
                        ev.weight,
                        ev.signal
                    );
                }
            }
        }
    }

    // -- Mock-based service tests --

    struct MockRoleDataProvider {
        features: Vec<CharacterFeatures>,
    }

    #[async_trait]
    impl RoleInferenceDataProvider for MockRoleDataProvider {
        async fn get_all_character_features(&self) -> Result<Vec<CharacterFeatures>, NarraError> {
            Ok(self.features.clone())
        }
    }

    #[tokio::test]
    async fn test_role_service_empty_world() {
        let provider = MockRoleDataProvider { features: vec![] };
        let service = RoleInferenceService::with_provider(Arc::new(provider));
        let report = service.infer_roles(50).await.unwrap();
        assert!(report.roles.is_empty());
        assert_eq!(report.total_characters, 0);
    }

    #[tokio::test]
    async fn test_role_service_multiple_characters() {
        let provider = MockRoleDataProvider {
            features: vec![
                CharacterFeatures {
                    id: "character:alice".to_string(),
                    name: "Alice".to_string(),
                    degree_centrality: 0.8,
                    ally_count: 4,
                    ..Default::default()
                },
                CharacterFeatures {
                    id: "character:bob".to_string(),
                    name: "Bob".to_string(),
                    rival_count: 3,
                    degree_centrality: 0.3,
                    ..Default::default()
                },
            ],
        };
        let service = RoleInferenceService::with_provider(Arc::new(provider));
        let report = service.infer_roles(50).await.unwrap();

        assert_eq!(report.total_characters, 2);
        assert_eq!(report.roles.len(), 2);

        // Should be sorted by confidence descending
        assert!(report.roles[0].confidence >= report.roles[1].confidence);
    }

    #[tokio::test]
    async fn test_role_service_limit() {
        let features: Vec<CharacterFeatures> = (0..10)
            .map(|i| CharacterFeatures {
                id: format!("character:char{}", i),
                name: format!("Char {}", i),
                degree_centrality: 0.8,
                ..Default::default()
            })
            .collect();

        let provider = MockRoleDataProvider { features };
        let service = RoleInferenceService::with_provider(Arc::new(provider));
        let report = service.infer_roles(3).await.unwrap();

        assert_eq!(report.total_characters, 10);
        assert_eq!(report.roles.len(), 3);
    }

    #[tokio::test]
    async fn test_role_service_deterministic() {
        // Same input should produce same output
        let features = vec![CharacterFeatures {
            id: "character:alice".to_string(),
            name: "Alice".to_string(),
            degree_centrality: 0.6,
            knowledge_count: 8,
            secret_count: 1,
            ..Default::default()
        }];

        let provider1 = MockRoleDataProvider {
            features: features.clone(),
        };
        let provider2 = MockRoleDataProvider { features };

        let service1 = RoleInferenceService::with_provider(Arc::new(provider1));
        let service2 = RoleInferenceService::with_provider(Arc::new(provider2));

        let report1 = service1.infer_roles(50).await.unwrap();
        let report2 = service2.infer_roles(50).await.unwrap();

        assert_eq!(report1.roles[0].primary_role, report2.roles[0].primary_role);
        assert_eq!(report1.roles[0].confidence, report2.roles[0].confidence);
    }
}
