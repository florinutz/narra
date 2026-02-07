//! Composite intelligence service that orchestrates multiple analytics services
//! into higher-level narrative insights.

use serde::Serialize;
use std::sync::Arc;
use surrealdb::engine::local::Db;
use surrealdb::Surreal;

use crate::models::knowledge::find_knowledge_conflicts;
use crate::services::{
    CentralityMetric, ClusteringService, EntityType, GraphAnalyticsService, InfluenceService,
    IronyService, KnowledgeAsymmetry,
};
use crate::NarraError;

/// Composite intelligence service for high-level narrative analysis.
///
/// Orchestrates multiple analytics services to produce combined insights
/// like situation reports, character dossiers, and scene planning.
pub struct CompositeIntelligenceService {
    db: Arc<Surreal<Db>>,
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// A summary of a knowledge conflict (BelievesWrongly).
#[derive(Debug, Clone, Serialize)]
pub struct ConflictSummary {
    pub target: String,
    pub character_id: String,
    pub certainty: String,
    pub truth_value: String,
}

/// A high-tension perception pair.
#[derive(Debug, Clone, Serialize)]
pub struct TensionPair {
    pub observer: String,
    pub target: String,
    pub tension_level: i32,
    pub feelings: Option<String>,
}

/// Full narrative situation report.
#[derive(Debug, Clone, Serialize)]
pub struct SituationReport {
    pub irony_highlights: Vec<KnowledgeAsymmetry>,
    pub knowledge_conflicts: Vec<ConflictSummary>,
    pub high_tension_pairs: Vec<TensionPair>,
    pub theme_count: usize,
    pub suggestions: Vec<String>,
}

/// How others perceive a character.
#[derive(Debug, Clone, Serialize)]
pub struct PerceptionSummary {
    pub observer: String,
    pub tension_level: Option<i32>,
    pub feelings: Option<String>,
}

/// Comprehensive dossier for a single character.
#[derive(Debug, Clone, Serialize)]
pub struct CharacterDossier {
    pub name: String,
    pub roles: Vec<String>,
    pub centrality_rank: Option<usize>,
    pub influence_reach: usize,
    pub knowledge_advantages: usize,
    pub knowledge_blind_spots: usize,
    pub false_beliefs: usize,
    pub avg_tension_toward_them: Option<f32>,
    pub key_perceptions: Vec<PerceptionSummary>,
    pub suggestions: Vec<String>,
}

/// Dynamics between a pair of characters for scene planning.
#[derive(Debug, Clone, Serialize)]
pub struct PairDynamic {
    pub character_a: String,
    pub character_b: String,
    pub asymmetries: usize,
    pub tension_level: Option<i32>,
    pub feelings: Option<String>,
    pub shared_scene_count: usize,
}

/// Scene planning result for a set of characters.
#[derive(Debug, Clone, Serialize)]
pub struct ScenePlan {
    pub characters: Vec<String>,
    pub pair_dynamics: Vec<PairDynamic>,
    pub total_irony_opportunities: usize,
    pub highest_tension_pair: Option<(String, String, i32)>,
    pub shared_history_scenes: usize,
    pub applicable_facts: Vec<String>,
    pub opportunities: Vec<String>,
}

// ---------------------------------------------------------------------------
// Internal query types
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct TensionRow {
    observer: String,
    target: String,
    observer_name: Option<String>,
    target_name: Option<String>,
    tension_level: i32,
    feelings: Option<String>,
}

#[derive(serde::Deserialize)]
struct PerceptionRow {
    observer: String,
    observer_name: Option<String>,
    tension_level: Option<i32>,
    feelings: Option<String>,
}

#[derive(serde::Deserialize)]
struct CharacterRow {
    name: String,
    roles: Vec<String>,
}

#[derive(serde::Deserialize)]
struct KnowledgeCountRow {
    count: usize,
}

#[derive(serde::Deserialize)]
struct FactTitleRow {
    title: String,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl CompositeIntelligenceService {
    pub fn new(db: Arc<Surreal<Db>>) -> Self {
        Self { db }
    }

    /// Generate a high-level narrative situation report.
    ///
    /// Combines irony, conflicts, tensions, and thematic clustering into one view.
    pub async fn situation_report(&self) -> Result<SituationReport, NarraError> {
        // 1. Irony highlights
        let irony_service = IronyService::new(self.db.clone());
        let irony_report = irony_service.generate_report(None, 3).await?;
        let mut irony_highlights = irony_report.asymmetries;
        irony_highlights.sort_by(|a, b| {
            b.dramatic_weight
                .partial_cmp(&a.dramatic_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        irony_highlights.truncate(5);

        // 2. Knowledge conflicts
        let raw_conflicts = find_knowledge_conflicts(&self.db).await?;
        let mut knowledge_conflicts = Vec::new();
        for conflict in &raw_conflicts {
            for state in &conflict.conflicting_states {
                knowledge_conflicts.push(ConflictSummary {
                    target: conflict.target.clone(),
                    character_id: state.character_id.clone(),
                    certainty: format!("{:?}", state.certainty),
                    truth_value: state
                        .truth_value
                        .as_deref()
                        .unwrap_or("unknown")
                        .to_string(),
                });
            }
        }

        // 3. High-tension pairs (tension >= 7)
        let high_tension_pairs = self.query_tension_pairs(7, 10).await?;

        // 4. Theme count
        let clustering_service = ClusteringService::new(self.db.clone());
        let theme_count = match clustering_service
            .discover_themes(
                vec![EntityType::Character, EntityType::Event, EntityType::Scene],
                Some(5),
            )
            .await
        {
            Ok(result) => result.clusters.len(),
            Err(_) => 0,
        };

        // 5. Generate suggestions
        let suggestions = generate_situation_suggestions(
            &irony_highlights,
            &knowledge_conflicts,
            &high_tension_pairs,
        );

        Ok(SituationReport {
            irony_highlights,
            knowledge_conflicts,
            high_tension_pairs,
            theme_count,
            suggestions,
        })
    }

    /// Generate a comprehensive dossier for a single character.
    pub async fn character_dossier(
        &self,
        character_id: &str,
    ) -> Result<CharacterDossier, NarraError> {
        let full_id = normalize_character_id(character_id);

        // 1. Basic info
        let (name, roles) = self.fetch_character_info(&full_id).await?;

        // 2. Centrality rank
        let analytics = GraphAnalyticsService::new(self.db.clone());
        let centrality_rank = match analytics
            .compute_centrality(None, vec![CentralityMetric::Degree], 100)
            .await
        {
            Ok(results) => results
                .iter()
                .position(|r| r.character_id == full_id)
                .map(|p| p + 1),
            Err(_) => None,
        };

        // 3. Influence reach
        let influence_service = InfluenceService::new(self.db.clone());
        let influence_reach = match influence_service.trace_propagation(&full_id, 3).await {
            Ok(result) => result.reachable_characters.len(),
            Err(_) => 0,
        };

        // 4. Irony report for this character
        let irony_service = IronyService::new(self.db.clone());
        let irony_report = match irony_service.generate_report(Some(&full_id), 0).await {
            Ok(r) => r,
            Err(_) => crate::services::IronyReport {
                focus: full_id.clone(),
                asymmetries: vec![],
                total_asymmetries: 0,
                high_signal_count: 0,
                narrative_opportunities: vec![],
            },
        };

        // Count advantages (this character knows, others don't) and blind spots (others know, this doesn't)
        let knowledge_advantages = irony_report
            .asymmetries
            .iter()
            .filter(|a| a.knowing_character_id == full_id)
            .count();
        let knowledge_blind_spots = irony_report
            .asymmetries
            .iter()
            .filter(|a| a.unknowing_character_id == full_id)
            .count();

        // 5. False beliefs count
        let false_beliefs = self.count_false_beliefs(&full_id).await?;

        // 6. Perceptions about this character
        let (avg_tension, key_perceptions) = self.fetch_perceptions_about(&full_id, 5).await?;

        // 7. Suggestions
        let suggestions = generate_dossier_suggestions(
            &name,
            centrality_rank,
            influence_reach,
            knowledge_advantages,
            knowledge_blind_spots,
            false_beliefs,
            &key_perceptions,
        );

        Ok(CharacterDossier {
            name,
            roles,
            centrality_rank,
            influence_reach,
            knowledge_advantages,
            knowledge_blind_spots,
            false_beliefs,
            avg_tension_toward_them: avg_tension,
            key_perceptions,
            suggestions,
        })
    }

    /// Plan a scene for a set of characters about to meet.
    pub async fn scene_prep(&self, character_ids: &[String]) -> Result<ScenePlan, NarraError> {
        let normalized: Vec<String> = character_ids
            .iter()
            .map(|id| normalize_character_id(id))
            .collect();

        let irony_service = IronyService::new(self.db.clone());

        let mut pair_dynamics = Vec::new();
        let mut total_irony = 0usize;
        let mut highest_tension: Option<(String, String, i32)> = None;
        let mut total_shared = 0usize;

        // For each pair
        for i in 0..normalized.len() {
            for j in (i + 1)..normalized.len() {
                let a = &normalized[i];
                let b = &normalized[j];

                // Asymmetries
                let asymmetries: Vec<KnowledgeAsymmetry> = irony_service
                    .detect_asymmetries(a, b)
                    .await
                    .unwrap_or_default();
                let asym_count = asymmetries.len();
                total_irony += asym_count;

                // Tension and feelings from perceives edge
                let (tension, feelings) = self.fetch_pair_tension(a, b).await;

                // Track highest tension
                if let Some(t) = tension {
                    match &highest_tension {
                        Some((_, _, current)) if t > *current => {
                            highest_tension = Some((a.clone(), b.clone(), t));
                        }
                        None => {
                            highest_tension = Some((a.clone(), b.clone(), t));
                        }
                        _ => {}
                    }
                }

                // Shared scenes
                let shared = self.count_shared_scenes(a, b).await;
                total_shared += shared;

                pair_dynamics.push(PairDynamic {
                    character_a: a.clone(),
                    character_b: b.clone(),
                    asymmetries: asym_count,
                    tension_level: tension,
                    feelings,
                    shared_scene_count: shared,
                });
            }
        }

        // Applicable facts
        let applicable_facts = self.fetch_applicable_facts(&normalized).await?;

        // Opportunities
        let opportunities = generate_scene_opportunities(&pair_dynamics, &applicable_facts);

        Ok(ScenePlan {
            characters: normalized,
            pair_dynamics,
            total_irony_opportunities: total_irony,
            highest_tension_pair: highest_tension,
            shared_history_scenes: total_shared,
            applicable_facts,
            opportunities,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    async fn query_tension_pairs(
        &self,
        min_tension: i32,
        limit: usize,
    ) -> Result<Vec<TensionPair>, NarraError> {
        let mut result = self
            .db
            .query(
                "SELECT type::string(in) AS observer, type::string(out) AS target, \
                 in.name AS observer_name, out.name AS target_name, \
                 tension_level, feelings \
                 FROM perceives \
                 WHERE tension_level IS NOT NONE AND tension_level >= $min_tension \
                 ORDER BY tension_level DESC \
                 LIMIT $limit",
            )
            .bind(("min_tension", min_tension))
            .bind(("limit", limit))
            .await
            .map_err(|e| NarraError::Database(e.to_string()))?;

        let rows: Vec<TensionRow> = result.take(0).unwrap_or_default();
        Ok(rows
            .into_iter()
            .map(|r| TensionPair {
                observer: r.observer_name.unwrap_or(r.observer),
                target: r.target_name.unwrap_or(r.target),
                tension_level: r.tension_level,
                feelings: r.feelings,
            })
            .collect())
    }

    async fn fetch_character_info(
        &self,
        full_id: &str,
    ) -> Result<(String, Vec<String>), NarraError> {
        let mut result = self
            .db
            .query(format!("SELECT name, roles FROM {}", full_id))
            .await
            .map_err(|e| NarraError::Database(e.to_string()))?;

        let row: Option<CharacterRow> = result.take(0).unwrap_or(None);
        match row {
            Some(r) => Ok((r.name, r.roles)),
            None => Err(NarraError::NotFound {
                entity_type: "character".to_string(),
                id: full_id.to_string(),
            }),
        }
    }

    async fn count_false_beliefs(&self, full_id: &str) -> Result<usize, NarraError> {
        let mut result = self
            .db
            .query(
                "SELECT count() AS count FROM knows \
                 WHERE in = $char_id AND certainty = 'BelievesWrongly' \
                 GROUP ALL",
            )
            .bind((
                "char_id",
                surrealdb::RecordId::from(
                    full_id.split_once(':').unwrap_or(("character", full_id)),
                ),
            ))
            .await
            .map_err(|e| NarraError::Database(e.to_string()))?;

        let row: Option<KnowledgeCountRow> = result.take(0).unwrap_or(None);
        Ok(row.map(|r| r.count).unwrap_or(0))
    }

    async fn fetch_perceptions_about(
        &self,
        full_id: &str,
        limit: usize,
    ) -> Result<(Option<f32>, Vec<PerceptionSummary>), NarraError> {
        let mut result = self
            .db
            .query(format!(
                "SELECT type::string(in) AS observer, in.name AS observer_name, \
                 tension_level, feelings \
                 FROM perceives \
                 WHERE out = {} \
                 ORDER BY tension_level DESC \
                 LIMIT {}",
                full_id, limit
            ))
            .await
            .map_err(|e| NarraError::Database(e.to_string()))?;

        let rows: Vec<PerceptionRow> = result.take(0).unwrap_or_default();

        let tensions: Vec<f32> = rows
            .iter()
            .filter_map(|r| r.tension_level.map(|t| t as f32))
            .collect();
        let avg = if tensions.is_empty() {
            None
        } else {
            Some(tensions.iter().sum::<f32>() / tensions.len() as f32)
        };

        let perceptions = rows
            .into_iter()
            .map(|r| PerceptionSummary {
                observer: r.observer_name.unwrap_or(r.observer),
                tension_level: r.tension_level,
                feelings: r.feelings,
            })
            .collect();

        Ok((avg, perceptions))
    }

    async fn fetch_pair_tension(&self, a: &str, b: &str) -> (Option<i32>, Option<String>) {
        // Check both directions, take the higher tension
        let query = format!(
            "SELECT tension_level, feelings FROM perceives \
             WHERE (in = {} AND out = {}) OR (in = {} AND out = {}) \
             ORDER BY tension_level DESC LIMIT 1",
            a, b, b, a
        );

        #[derive(serde::Deserialize)]
        struct PairRow {
            tension_level: Option<i32>,
            feelings: Option<String>,
        }

        match self.db.query(&query).await {
            Ok(mut r) => {
                let row: Option<PairRow> = r.take(0).unwrap_or(None);
                match row {
                    Some(pr) => (pr.tension_level, pr.feelings),
                    None => (None, None),
                }
            }
            Err(_) => (None, None),
        }
    }

    async fn count_shared_scenes(&self, a: &str, b: &str) -> usize {
        let query = format!(
            "SELECT count() AS count FROM (SELECT out FROM participates_in WHERE in = {}) \
             WHERE out IN (SELECT VALUE out FROM participates_in WHERE in = {}) \
             GROUP ALL",
            a, b
        );

        match self.db.query(&query).await {
            Ok(mut r) => {
                let row: Option<KnowledgeCountRow> = r.take(0).unwrap_or(None);
                row.map(|r| r.count).unwrap_or(0)
            }
            Err(_) => 0,
        }
    }

    async fn fetch_applicable_facts(
        &self,
        character_ids: &[String],
    ) -> Result<Vec<String>, NarraError> {
        if character_ids.is_empty() {
            return Ok(vec![]);
        }

        // Build an array of character IDs for the query
        let id_list: Vec<String> = character_ids.iter().map(|id| id.to_string()).collect();
        let array_str = id_list.join(", ");
        let query = format!(
            "SELECT title FROM universe_fact \
             WHERE id IN (SELECT VALUE fact FROM fact_applies WHERE entity IN [{}])",
            array_str
        );

        let mut result = self
            .db
            .query(&query)
            .await
            .map_err(|e| NarraError::Database(e.to_string()))?;

        let rows: Vec<FactTitleRow> = result.take(0).unwrap_or_default();
        Ok(rows.into_iter().map(|r| r.title).collect())
    }
}

// ---------------------------------------------------------------------------
// Pure suggestion generators (testable)
// ---------------------------------------------------------------------------

fn generate_situation_suggestions(
    irony: &[KnowledgeAsymmetry],
    conflicts: &[ConflictSummary],
    tensions: &[TensionPair],
) -> Vec<String> {
    let mut suggestions = Vec::new();

    if !irony.is_empty() {
        let top = &irony[0];
        suggestions.push(format!(
            "High dramatic irony: {} knows something {} doesn't about {} — consider a reveal scene",
            top.knowing_character_name, top.unknowing_character_name, top.about
        ));
    }

    if !conflicts.is_empty() {
        suggestions.push(format!(
            "{} character(s) hold false beliefs — potential confrontation or discovery moments",
            conflicts.len()
        ));
    }

    if !tensions.is_empty() {
        let top = &tensions[0];
        suggestions.push(format!(
            "Highest tension ({}) between {} and {} — ripe for conflict escalation",
            top.tension_level, top.observer, top.target
        ));
    }

    if irony.is_empty() && conflicts.is_empty() && tensions.is_empty() {
        suggestions.push("The narrative is relatively stable — consider introducing new secrets, misunderstandings, or conflicting goals".to_string());
    }

    suggestions
}

fn generate_dossier_suggestions(
    name: &str,
    centrality_rank: Option<usize>,
    influence_reach: usize,
    knowledge_advantages: usize,
    knowledge_blind_spots: usize,
    false_beliefs: usize,
    perceptions: &[PerceptionSummary],
) -> Vec<String> {
    let mut suggestions = Vec::new();

    if let Some(rank) = centrality_rank {
        if rank <= 3 {
            suggestions.push(format!(
                "{} is a central figure (rank #{}) — their actions have wide narrative impact",
                name, rank
            ));
        } else if rank > 10 {
            suggestions.push(format!(
                "{} is peripheral (rank #{}) — consider strengthening connections or making isolation a plot point",
                name, rank
            ));
        }
    }

    if influence_reach == 0 {
        suggestions.push(format!(
            "{} has no influence paths — they're isolated from the information network",
            name
        ));
    }

    if knowledge_blind_spots > knowledge_advantages {
        suggestions.push(format!(
            "{} has more blind spots ({}) than advantages ({}) — vulnerable to surprises",
            name, knowledge_blind_spots, knowledge_advantages
        ));
    } else if knowledge_advantages > 0 {
        suggestions.push(format!(
            "{} holds {} knowledge advantages — potential for strategic reveals or leverage",
            name, knowledge_advantages
        ));
    }

    if false_beliefs > 0 {
        suggestions.push(format!(
            "{} holds {} false belief(s) — each is a potential turning point when corrected",
            name, false_beliefs
        ));
    }

    let high_tension: Vec<_> = perceptions
        .iter()
        .filter(|p| p.tension_level.map(|t| t >= 7).unwrap_or(false))
        .collect();
    if !high_tension.is_empty() {
        suggestions.push(format!(
            "{} observer(s) have high tension toward {} — unresolved conflicts ahead",
            high_tension.len(),
            name
        ));
    }

    suggestions
}

fn generate_scene_opportunities(
    dynamics: &[PairDynamic],
    applicable_facts: &[String],
) -> Vec<String> {
    let mut opportunities = Vec::new();

    // Pairs with asymmetries but no shared scenes = first meeting potential
    for d in dynamics {
        if d.asymmetries > 0 && d.shared_scene_count == 0 {
            opportunities.push(format!(
                "{} and {} have never shared a scene but have {} knowledge asymmetries — first meeting would be dramatic",
                d.character_a, d.character_b, d.asymmetries
            ));
        }
    }

    // High tension pairs
    for d in dynamics {
        if d.tension_level.map(|t| t >= 7).unwrap_or(false) {
            opportunities.push(format!(
                "High tension ({}) between {} and {} — confrontation or reconciliation moment",
                d.tension_level.unwrap_or(0),
                d.character_a,
                d.character_b
            ));
        }
    }

    if !applicable_facts.is_empty() {
        opportunities.push(format!(
            "{} universe fact(s) apply to these characters — consider fact enforcement in the scene",
            applicable_facts.len()
        ));
    }

    opportunities
}

/// Combine pairs from N characters.
pub fn pair_count(n: usize) -> usize {
    if n < 2 {
        0
    } else {
        n * (n - 1) / 2
    }
}

fn normalize_character_id(id: &str) -> String {
    if id.contains(':') {
        id.to_string()
    } else {
        format!("character:{}", id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_count() {
        assert_eq!(pair_count(0), 0);
        assert_eq!(pair_count(1), 0);
        assert_eq!(pair_count(2), 1);
        assert_eq!(pair_count(3), 3);
        assert_eq!(pair_count(4), 6);
        assert_eq!(pair_count(5), 10);
    }

    #[test]
    fn test_normalize_character_id() {
        assert_eq!(normalize_character_id("alice"), "character:alice");
        assert_eq!(normalize_character_id("character:alice"), "character:alice");
        assert_eq!(normalize_character_id("location:castle"), "location:castle");
    }

    #[test]
    fn test_situation_suggestions_empty_world() {
        let suggestions = generate_situation_suggestions(&[], &[], &[]);
        assert_eq!(suggestions.len(), 1);
        assert!(suggestions[0].contains("stable"));
    }

    #[test]
    fn test_situation_suggestions_with_irony() {
        let irony = vec![KnowledgeAsymmetry {
            knowing_character_id: "character:alice".to_string(),
            knowing_character_name: "Alice".to_string(),
            unknowing_character_id: "character:bob".to_string(),
            unknowing_character_name: "Bob".to_string(),
            fact: "the secret".to_string(),
            learning_method: "Witnessed".to_string(),
            certainty: "Knows".to_string(),
            scenes_since: 3,
            signal_strength: "high".to_string(),
            about: "the betrayal".to_string(),
            dramatic_weight: 8.0,
            tension_level: Some(7),
            feelings: Some("suspicious".to_string()),
            history_notes: None,
        }];
        let suggestions = generate_situation_suggestions(&irony, &[], &[]);
        assert!(suggestions[0].contains("Alice"));
        assert!(suggestions[0].contains("Bob"));
        assert!(suggestions[0].contains("reveal"));
    }

    #[test]
    fn test_situation_suggestions_with_conflicts() {
        let conflicts = vec![ConflictSummary {
            target: "knowledge:secret".to_string(),
            character_id: "character:bob".to_string(),
            certainty: "BelievesWrongly".to_string(),
            truth_value: "false".to_string(),
        }];
        let suggestions = generate_situation_suggestions(&[], &conflicts, &[]);
        assert!(suggestions[0].contains("false beliefs"));
    }

    #[test]
    fn test_dossier_suggestions_central_character() {
        let suggestions = generate_dossier_suggestions("Alice", Some(1), 5, 3, 1, 0, &[]);
        assert!(suggestions.iter().any(|s| s.contains("central")));
        assert!(suggestions.iter().any(|s| s.contains("advantages")));
    }

    #[test]
    fn test_dossier_suggestions_peripheral_character() {
        let suggestions = generate_dossier_suggestions("Bob", Some(15), 0, 0, 3, 2, &[]);
        assert!(suggestions.iter().any(|s| s.contains("peripheral")));
        assert!(suggestions.iter().any(|s| s.contains("isolated")));
        assert!(suggestions.iter().any(|s| s.contains("blind spots")));
        assert!(suggestions.iter().any(|s| s.contains("false belief")));
    }

    #[test]
    fn test_scene_opportunities_first_meeting() {
        let dynamics = vec![PairDynamic {
            character_a: "character:alice".to_string(),
            character_b: "character:bob".to_string(),
            asymmetries: 2,
            tension_level: None,
            feelings: None,
            shared_scene_count: 0,
        }];
        let opportunities = generate_scene_opportunities(&dynamics, &[]);
        assert!(opportunities[0].contains("never shared a scene"));
    }

    #[test]
    fn test_scene_opportunities_high_tension() {
        let dynamics = vec![PairDynamic {
            character_a: "character:alice".to_string(),
            character_b: "character:bob".to_string(),
            asymmetries: 0,
            tension_level: Some(9),
            feelings: Some("hostile".to_string()),
            shared_scene_count: 3,
        }];
        let opportunities = generate_scene_opportunities(&dynamics, &[]);
        assert!(opportunities[0].contains("High tension"));
    }
}
