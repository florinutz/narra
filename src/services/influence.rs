//! Influence propagation service for tracing information flow through character networks.
//!
//! Models how knowledge could spread through directed relationship paths.
//! Respects asymmetric relationships - if Alice trusts Bob, that doesn't mean
//! Bob trusts Alice. Only follows OUTGOING perceives edges for propagation.

use async_trait::async_trait;
use serde::Serialize;
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use surrealdb::{engine::local::Db, Surreal};

use crate::NarraError;

/// A single step in an influence propagation path.
#[derive(Debug, Clone, Serialize)]
pub struct InfluenceStep {
    pub character_id: String,
    pub character_name: String,
    pub relationship_type: String,
    pub depth: usize,
    /// Tension level from the perceives edge leading to this step
    pub tension_level: Option<i32>,
}

/// A complete propagation path from source to a reachable character.
#[derive(Debug, Clone, Serialize)]
pub struct InfluencePath {
    pub steps: Vec<InfluenceStep>,
    pub total_hops: usize,
    pub path_strength: String,
    /// Numeric strength (0.0-1.0) factoring in hops, tension, and rel_type
    pub numeric_strength: f32,
}

/// Result of an influence propagation trace.
#[derive(Debug, Clone, Serialize)]
pub struct PropagationResult {
    pub source_character: String,
    pub knowledge_summary: String,
    pub reachable_characters: Vec<InfluencePath>,
    pub unreachable_characters: Vec<String>,
}

// ---------------------------------------------------------------------------
// Pure functions
// ---------------------------------------------------------------------------

/// Classify path strength based on hop count (backward-compat label).
pub(crate) fn classify_path_strength(hops: usize) -> &'static str {
    match hops {
        1 => "direct",
        2 => "likely",
        _ => "possible",
    }
}

/// Compute path strength factoring in hops, tension, and relationship types.
/// Returns `(label, numeric_strength)` where numeric_strength is 0.0-1.0.
pub(crate) fn compute_path_strength(steps: &[InfluenceStep]) -> (String, f32) {
    let hops = steps.len().saturating_sub(1);
    let label = classify_path_strength(hops).to_string();

    // Base from hop count
    let mut strength: f32 = match hops {
        1 => 1.0,
        2 => 0.6,
        _ => 0.3,
    };

    // Apply per-edge modifiers (skip the source step at index 0)
    for step in steps.iter().skip(1) {
        // High tension reduces strength (enemies less likely to share)
        if let Some(t) = step.tension_level {
            if t >= 7 {
                strength *= 0.7; // -30% for high tension
            }
        }

        // Relationship type affects flow
        let rel_lower = step.relationship_type.to_lowercase();
        let rel_multiplier = if rel_lower.contains("mentor") || rel_lower.contains("family") {
            1.0 // reliable channels
        } else if rel_lower.contains("ally") || rel_lower.contains("friend") {
            0.9
        } else if rel_lower.contains("professional") {
            0.7
        } else if rel_lower.contains("rival") {
            0.4
        } else {
            0.85 // default for unknown types
        };
        strength *= rel_multiplier;
    }

    strength = strength.clamp(0.0, 1.0);
    (label, strength)
}

// ---------------------------------------------------------------------------
// Data provider trait
// ---------------------------------------------------------------------------

/// An outgoing perception edge (target ID + relationship types).
#[derive(Debug, Clone)]
pub struct PerceptionEdgeInfo {
    pub target_id: String,
    pub rel_types: Vec<String>,
    /// Tension level from the perceives edge (0-10)
    pub tension_level: Option<i32>,
    /// Relationship subtype for finer classification
    pub subtype: Option<String>,
}

/// Data access abstraction for the influence service.
#[async_trait]
pub trait InfluenceDataProvider: Send + Sync {
    async fn get_outgoing_edges(
        &self,
        character_key: &str,
    ) -> Result<Vec<PerceptionEdgeInfo>, NarraError>;
    async fn get_character_name(&self, key: &str) -> Result<String, NarraError>;
    async fn get_all_character_ids(&self) -> Result<Vec<String>, NarraError>;
}

/// SurrealDB implementation of InfluenceDataProvider.
pub struct SurrealInfluenceDataProvider {
    db: Arc<Surreal<Db>>,
}

impl SurrealInfluenceDataProvider {
    pub fn new(db: Arc<Surreal<Db>>) -> Self {
        Self { db }
    }
}

/// Edge structure returned from perceives queries.
#[derive(Debug, serde::Deserialize)]
struct PerceptionEdge {
    target: surrealdb::sql::Thing,
    rel_types: Vec<String>,
    tension_level: Option<i32>,
    subtype: Option<String>,
}

/// Character info for name resolution.
#[derive(Debug, serde::Deserialize)]
struct CharacterInfo {
    name: String,
}

#[async_trait]
impl InfluenceDataProvider for SurrealInfluenceDataProvider {
    async fn get_outgoing_edges(
        &self,
        character_key: &str,
    ) -> Result<Vec<PerceptionEdgeInfo>, NarraError> {
        let query = format!(
            "SELECT out AS target, rel_types, tension_level, subtype FROM perceives WHERE in = character:{}",
            character_key
        );
        let mut result = self.db.query(&query).await?;
        let edges: Vec<PerceptionEdge> = result.take(0).unwrap_or_default();

        Ok(edges
            .into_iter()
            .map(|e| PerceptionEdgeInfo {
                target_id: e.target.to_string(),
                rel_types: e.rel_types,
                tension_level: e.tension_level,
                subtype: e.subtype,
            })
            .collect())
    }

    async fn get_character_name(&self, key: &str) -> Result<String, NarraError> {
        let char_ref = surrealdb::RecordId::from(("character", key));
        let mut result = self
            .db
            .query("SELECT name FROM ONLY $char_ref")
            .bind(("char_ref", char_ref))
            .await?;
        let chars: Vec<CharacterInfo> = result.take(0)?;

        chars
            .into_iter()
            .next()
            .map(|c| c.name)
            .ok_or_else(|| NarraError::Validation(format!("Character not found: {}", key)))
    }

    async fn get_all_character_ids(&self) -> Result<Vec<String>, NarraError> {
        let mut result = self.db.query("SELECT VALUE id FROM character").await?;
        let ids: Vec<surrealdb::sql::Thing> = result.take(0)?;
        Ok(ids.into_iter().map(|id| id.to_string()).collect())
    }
}

// ---------------------------------------------------------------------------
// InfluenceService
// ---------------------------------------------------------------------------

/// Service for tracing influence propagation through character networks.
pub struct InfluenceService {
    data: Arc<dyn InfluenceDataProvider>,
}

impl InfluenceService {
    pub fn new(db: Arc<Surreal<Db>>) -> Self {
        Self {
            data: Arc::new(SurrealInfluenceDataProvider::new(db)),
        }
    }

    pub fn with_provider(data: Arc<dyn InfluenceDataProvider>) -> Self {
        Self { data }
    }

    /// Trace how influence/information could propagate from a character through the network.
    pub async fn trace_propagation(
        &self,
        from_character_id: &str,
        max_depth: usize,
    ) -> Result<PropagationResult, NarraError> {
        let source_key = from_character_id
            .split(':')
            .nth(1)
            .unwrap_or(from_character_id);

        let reachable_paths = self.bfs_directed_propagation(source_key, max_depth).await?;

        let all_characters = self.data.get_all_character_ids().await?;
        let reachable_ids: HashSet<String> = reachable_paths
            .iter()
            .map(|path| {
                path.steps
                    .last()
                    .map(|s| s.character_id.clone())
                    .unwrap_or_default()
            })
            .collect();

        let mut unreachable: Vec<String> = all_characters
            .into_iter()
            .filter(|id| {
                let key = id.split(':').nth(1).unwrap_or(id);
                key != source_key && !reachable_ids.contains(id)
            })
            .collect();
        unreachable.sort();

        Ok(PropagationResult {
            source_character: format!("character:{}", source_key),
            knowledge_summary: String::new(),
            reachable_characters: reachable_paths,
            unreachable_characters: unreachable,
        })
    }

    /// Trace how specific knowledge could propagate from a character.
    pub async fn trace_knowledge_propagation(
        &self,
        from_character_id: &str,
        knowledge_fact: &str,
        max_depth: usize,
    ) -> Result<PropagationResult, NarraError> {
        let mut result = self.trace_propagation(from_character_id, max_depth).await?;
        result.knowledge_summary = knowledge_fact.to_string();
        Ok(result)
    }

    /// BFS traversal following OUTGOING edges only (directed propagation).
    async fn bfs_directed_propagation(
        &self,
        source_key: &str,
        max_depth: usize,
    ) -> Result<Vec<InfluencePath>, NarraError> {
        let mut paths: Vec<InfluencePath> = Vec::new();
        let mut queue: VecDeque<(String, Vec<InfluenceStep>, HashSet<String>, usize)> =
            VecDeque::new();

        let source_name = self.data.get_character_name(source_key).await?;

        let mut initial_visited = HashSet::new();
        initial_visited.insert(source_key.to_string());

        queue.push_back((
            source_key.to_string(),
            vec![InfluenceStep {
                character_id: format!("character:{}", source_key),
                character_name: source_name,
                relationship_type: "source".to_string(),
                depth: 0,
                tension_level: None,
            }],
            initial_visited,
            0,
        ));

        while let Some((current_id, path, visited, depth)) = queue.pop_front() {
            if depth >= max_depth {
                if depth > 0 {
                    paths.push(Self::build_influence_path(path));
                }
                continue;
            }

            let outgoing = self.data.get_outgoing_edges(&current_id).await?;

            if outgoing.is_empty() && depth > 0 {
                paths.push(Self::build_influence_path(path.clone()));
                continue;
            }

            for edge in outgoing {
                let target_full = &edge.target_id;
                let target_key = target_full
                    .split(':')
                    .nth(1)
                    .unwrap_or(target_full)
                    .to_string();

                if visited.contains(&target_key) {
                    continue;
                }

                if target_key == current_id {
                    continue;
                }

                let rel_type = edge
                    .rel_types
                    .first()
                    .map(|s| s.as_str())
                    .unwrap_or("relationship");

                let target_name = match self.data.get_character_name(&target_key).await {
                    Ok(name) => name,
                    Err(_) => continue,
                };

                let mut new_path = path.clone();
                new_path.push(InfluenceStep {
                    character_id: format!("character:{}", target_key),
                    character_name: target_name,
                    relationship_type: rel_type.to_string(),
                    depth: depth + 1,
                    tension_level: edge.tension_level,
                });

                let mut new_visited = visited.clone();
                new_visited.insert(target_key.clone());

                queue.push_back((target_key, new_path, new_visited, depth + 1));

                if paths.len() >= 50 {
                    break;
                }
            }

            if paths.len() >= 50 {
                break;
            }
        }

        paths.sort_by_key(|p| p.total_hops);

        Ok(paths)
    }

    /// Build an InfluencePath from steps, calculating strength.
    fn build_influence_path(steps: Vec<InfluenceStep>) -> InfluencePath {
        let total_hops = steps.len().saturating_sub(1);
        let (label, numeric) = compute_path_strength(&steps);

        InfluencePath {
            steps,
            total_hops,
            path_strength: label,
            numeric_strength: numeric,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Pure function tests --

    #[test]
    fn test_classify_path_strength_direct() {
        assert_eq!(classify_path_strength(1), "direct");
    }

    #[test]
    fn test_classify_path_strength_likely() {
        assert_eq!(classify_path_strength(2), "likely");
    }

    #[test]
    fn test_classify_path_strength_possible() {
        assert_eq!(classify_path_strength(0), "possible");
        assert_eq!(classify_path_strength(3), "possible");
        assert_eq!(classify_path_strength(10), "possible");
    }

    // -- Mock-based service tests --

    struct MockInfluenceDataProvider {
        edges: std::collections::HashMap<String, Vec<PerceptionEdgeInfo>>,
        names: std::collections::HashMap<String, String>,
        all_ids: Vec<String>,
    }

    #[async_trait]
    impl InfluenceDataProvider for MockInfluenceDataProvider {
        async fn get_outgoing_edges(
            &self,
            character_key: &str,
        ) -> Result<Vec<PerceptionEdgeInfo>, NarraError> {
            Ok(self.edges.get(character_key).cloned().unwrap_or_default())
        }

        async fn get_character_name(&self, key: &str) -> Result<String, NarraError> {
            self.names
                .get(key)
                .cloned()
                .ok_or_else(|| NarraError::Validation(format!("Character not found: {}", key)))
        }

        async fn get_all_character_ids(&self) -> Result<Vec<String>, NarraError> {
            Ok(self.all_ids.clone())
        }
    }

    fn make_edge(target: &str, rel_type: &str) -> PerceptionEdgeInfo {
        PerceptionEdgeInfo {
            target_id: format!("character:{}", target),
            rel_types: vec![rel_type.to_string()],
            tension_level: None,
            subtype: None,
        }
    }

    fn make_edge_with_tension(target: &str, rel_type: &str, tension: i32) -> PerceptionEdgeInfo {
        PerceptionEdgeInfo {
            target_id: format!("character:{}", target),
            rel_types: vec![rel_type.to_string()],
            tension_level: Some(tension),
            subtype: None,
        }
    }

    #[tokio::test]
    async fn test_influence_linear_chain() {
        // A -> B -> C
        let mut edges = std::collections::HashMap::new();
        edges.insert("alice".to_string(), vec![make_edge("bob", "trusts")]);
        edges.insert("bob".to_string(), vec![make_edge("charlie", "trusts")]);

        let mut names = std::collections::HashMap::new();
        names.insert("alice".to_string(), "Alice".to_string());
        names.insert("bob".to_string(), "Bob".to_string());
        names.insert("charlie".to_string(), "Charlie".to_string());

        let provider = MockInfluenceDataProvider {
            edges,
            names,
            all_ids: vec![
                "character:alice".to_string(),
                "character:bob".to_string(),
                "character:charlie".to_string(),
            ],
        };
        let service = InfluenceService::with_provider(Arc::new(provider));

        let result = service
            .trace_propagation("character:alice", 3)
            .await
            .unwrap();
        assert_eq!(result.source_character, "character:alice");
        // BFS records terminal paths — charlie is the terminal of the A->B->C chain
        assert!(!result.reachable_characters.is_empty());
        // The path to charlie passes through bob (2 hops)
        let charlie_path = result
            .reachable_characters
            .iter()
            .find(|p| p.steps.last().map(|s| s.character_name.as_str()) == Some("Charlie"));
        assert!(charlie_path.is_some(), "Charlie should be reachable");
        assert_eq!(charlie_path.unwrap().total_hops, 2);
    }

    #[tokio::test]
    async fn test_influence_disconnected() {
        // A -> B, C is isolated
        let mut edges = std::collections::HashMap::new();
        edges.insert("alice".to_string(), vec![make_edge("bob", "trusts")]);

        let mut names = std::collections::HashMap::new();
        names.insert("alice".to_string(), "Alice".to_string());
        names.insert("bob".to_string(), "Bob".to_string());
        names.insert("charlie".to_string(), "Charlie".to_string());

        let provider = MockInfluenceDataProvider {
            edges,
            names,
            all_ids: vec![
                "character:alice".to_string(),
                "character:bob".to_string(),
                "character:charlie".to_string(),
            ],
        };
        let service = InfluenceService::with_provider(Arc::new(provider));

        let result = service
            .trace_propagation("character:alice", 3)
            .await
            .unwrap();
        assert_eq!(result.unreachable_characters, vec!["character:charlie"]);
    }

    #[tokio::test]
    async fn test_influence_max_depth_cutoff() {
        // A -> B -> C -> D, but max_depth = 1
        let mut edges = std::collections::HashMap::new();
        edges.insert("alice".to_string(), vec![make_edge("bob", "trusts")]);
        edges.insert("bob".to_string(), vec![make_edge("charlie", "trusts")]);
        edges.insert("charlie".to_string(), vec![make_edge("dave", "trusts")]);

        let mut names = std::collections::HashMap::new();
        names.insert("alice".to_string(), "Alice".to_string());
        names.insert("bob".to_string(), "Bob".to_string());
        names.insert("charlie".to_string(), "Charlie".to_string());
        names.insert("dave".to_string(), "Dave".to_string());

        let provider = MockInfluenceDataProvider {
            edges,
            names,
            all_ids: vec![
                "character:alice".to_string(),
                "character:bob".to_string(),
                "character:charlie".to_string(),
                "character:dave".to_string(),
            ],
        };
        let service = InfluenceService::with_provider(Arc::new(provider));

        let result = service
            .trace_propagation("character:alice", 1)
            .await
            .unwrap();
        // Only Bob should be reachable at depth 1
        let reachable_names: Vec<String> = result
            .reachable_characters
            .iter()
            .filter_map(|p| p.steps.last().map(|s| s.character_name.clone()))
            .collect();
        assert!(reachable_names.contains(&"Bob".to_string()));
        assert!(!reachable_names.contains(&"Charlie".to_string()));
    }

    #[tokio::test]
    async fn test_influence_cycle_prevention() {
        // A -> B -> C -> A (pure cycle). The BFS should terminate (not infinite loop)
        // and not panic. The BFS only records paths for leaf nodes (no outgoing edges)
        // or depth-limited nodes, so a pure cycle where every node has outgoing edges
        // (even if filtered by visited set) may record zero paths.
        let mut edges = std::collections::HashMap::new();
        edges.insert("alice".to_string(), vec![make_edge("bob", "trusts")]);
        edges.insert("bob".to_string(), vec![make_edge("charlie", "trusts")]);
        edges.insert("charlie".to_string(), vec![make_edge("alice", "trusts")]);

        let mut names = std::collections::HashMap::new();
        names.insert("alice".to_string(), "Alice".to_string());
        names.insert("bob".to_string(), "Bob".to_string());
        names.insert("charlie".to_string(), "Charlie".to_string());

        let provider = MockInfluenceDataProvider {
            edges,
            names,
            all_ids: vec![
                "character:alice".to_string(),
                "character:bob".to_string(),
                "character:charlie".to_string(),
            ],
        };
        let service = InfluenceService::with_provider(Arc::new(provider));

        // Should not infinite loop — terminates because visited set prevents re-visiting
        let result = service
            .trace_propagation("character:alice", 5)
            .await
            .unwrap();
        assert_eq!(result.source_character, "character:alice");
        // The BFS completes without panic (main assertion: it terminates)
    }

    #[tokio::test]
    async fn test_influence_hub_and_spoke() {
        // Alice -> Bob, Alice -> Charlie, Alice -> Dave
        let mut edges = std::collections::HashMap::new();
        edges.insert(
            "alice".to_string(),
            vec![
                make_edge("bob", "trusts"),
                make_edge("charlie", "trusts"),
                make_edge("dave", "trusts"),
            ],
        );

        let mut names = std::collections::HashMap::new();
        names.insert("alice".to_string(), "Alice".to_string());
        names.insert("bob".to_string(), "Bob".to_string());
        names.insert("charlie".to_string(), "Charlie".to_string());
        names.insert("dave".to_string(), "Dave".to_string());

        let provider = MockInfluenceDataProvider {
            edges,
            names,
            all_ids: vec![
                "character:alice".to_string(),
                "character:bob".to_string(),
                "character:charlie".to_string(),
                "character:dave".to_string(),
            ],
        };
        let service = InfluenceService::with_provider(Arc::new(provider));

        let result = service
            .trace_propagation("character:alice", 3)
            .await
            .unwrap();
        assert_eq!(result.reachable_characters.len(), 3);
        // All paths should be "direct" (1 hop)
        for path in &result.reachable_characters {
            assert_eq!(path.total_hops, 1);
            assert_eq!(path.path_strength, "direct");
        }
    }

    #[test]
    fn test_compute_path_strength_direct() {
        let steps = vec![
            InfluenceStep {
                character_id: "character:alice".to_string(),
                character_name: "Alice".to_string(),
                relationship_type: "source".to_string(),
                depth: 0,
                tension_level: None,
            },
            InfluenceStep {
                character_id: "character:bob".to_string(),
                character_name: "Bob".to_string(),
                relationship_type: "trusts".to_string(),
                depth: 1,
                tension_level: None,
            },
        ];
        let (label, strength) = compute_path_strength(&steps);
        assert_eq!(label, "direct");
        assert!((strength - 0.85).abs() < 0.01, "default rel type = 0.85x");
    }

    #[test]
    fn test_compute_path_strength_high_tension_reduces() {
        let steps = vec![
            InfluenceStep {
                character_id: "character:alice".to_string(),
                character_name: "Alice".to_string(),
                relationship_type: "source".to_string(),
                depth: 0,
                tension_level: None,
            },
            InfluenceStep {
                character_id: "character:bob".to_string(),
                character_name: "Bob".to_string(),
                relationship_type: "trusts".to_string(),
                depth: 1,
                tension_level: Some(9),
            },
        ];
        let (_, strength) = compute_path_strength(&steps);
        // 1.0 (base) * 0.7 (tension) * 0.85 (default rel) = 0.595
        assert!(
            strength < 0.6,
            "high tension should reduce strength: {}",
            strength
        );
    }

    #[test]
    fn test_compute_path_strength_rivalry_reduces() {
        let steps = vec![
            InfluenceStep {
                character_id: "character:alice".to_string(),
                character_name: "Alice".to_string(),
                relationship_type: "source".to_string(),
                depth: 0,
                tension_level: None,
            },
            InfluenceStep {
                character_id: "character:bob".to_string(),
                character_name: "Bob".to_string(),
                relationship_type: "rivalry".to_string(),
                depth: 1,
                tension_level: None,
            },
        ];
        let (_, strength) = compute_path_strength(&steps);
        // 1.0 (base) * 0.4 (rivalry) = 0.4
        assert!(
            (strength - 0.4).abs() < 0.01,
            "rivalry should reduce to 0.4: {}",
            strength
        );
    }

    #[test]
    fn test_compute_path_strength_multiple_high_tension_edges() {
        // Two consecutive high-tension edges should multiply: 0.7 * 0.7 = 0.49x
        let steps = vec![
            InfluenceStep {
                character_id: "character:alice".to_string(),
                character_name: "Alice".to_string(),
                relationship_type: "source".to_string(),
                depth: 0,
                tension_level: None,
            },
            InfluenceStep {
                character_id: "character:bob".to_string(),
                character_name: "Bob".to_string(),
                relationship_type: "mentor".to_string(),
                depth: 1,
                tension_level: Some(8), // high tension
            },
            InfluenceStep {
                character_id: "character:charlie".to_string(),
                character_name: "Charlie".to_string(),
                relationship_type: "mentor".to_string(),
                depth: 2,
                tension_level: Some(9), // high tension
            },
        ];
        let (label, strength) = compute_path_strength(&steps);
        assert_eq!(label, "likely"); // 2 hops
                                     // Base 0.6 * 0.7 (tension) * 1.0 (mentor) * 0.7 (tension) * 1.0 (mentor) = 0.294
        assert!(
            strength < 0.3,
            "Multiple high-tension edges should compound: {}",
            strength
        );
    }

    #[test]
    fn test_compute_path_strength_mixed_relationship_types() {
        // mentor → professional → rivalry: reliability degrades along the chain
        let steps = vec![
            InfluenceStep {
                character_id: "character:alice".to_string(),
                character_name: "Alice".to_string(),
                relationship_type: "source".to_string(),
                depth: 0,
                tension_level: None,
            },
            InfluenceStep {
                character_id: "character:bob".to_string(),
                character_name: "Bob".to_string(),
                relationship_type: "mentor".to_string(), // 1.0x
                depth: 1,
                tension_level: None,
            },
            InfluenceStep {
                character_id: "character:charlie".to_string(),
                character_name: "Charlie".to_string(),
                relationship_type: "professional".to_string(), // 0.7x
                depth: 2,
                tension_level: None,
            },
            InfluenceStep {
                character_id: "character:dave".to_string(),
                character_name: "Dave".to_string(),
                relationship_type: "rivalry".to_string(), // 0.4x
                depth: 3,
                tension_level: None,
            },
        ];
        let (label, strength) = compute_path_strength(&steps);
        assert_eq!(label, "possible"); // 3 hops
                                       // Base 0.3 * 1.0 (mentor) * 0.7 (professional) * 0.4 (rivalry) = 0.084
        assert!(
            strength < 0.1,
            "Mixed rel types with rivalry should severely degrade: {}",
            strength
        );
        assert!(strength > 0.0, "Should still be positive: {}", strength);
    }

    #[test]
    fn test_compute_path_strength_none_tension_same_as_zero() {
        // None tension should not affect strength (same as no tension)
        let steps_none = vec![
            InfluenceStep {
                character_id: "character:alice".to_string(),
                character_name: "Alice".to_string(),
                relationship_type: "source".to_string(),
                depth: 0,
                tension_level: None,
            },
            InfluenceStep {
                character_id: "character:bob".to_string(),
                character_name: "Bob".to_string(),
                relationship_type: "friend".to_string(),
                depth: 1,
                tension_level: None,
            },
        ];
        let steps_low = vec![
            InfluenceStep {
                character_id: "character:alice".to_string(),
                character_name: "Alice".to_string(),
                relationship_type: "source".to_string(),
                depth: 0,
                tension_level: None,
            },
            InfluenceStep {
                character_id: "character:bob".to_string(),
                character_name: "Bob".to_string(),
                relationship_type: "friend".to_string(),
                depth: 1,
                tension_level: Some(3), // below threshold
            },
        ];
        let (_, strength_none) = compute_path_strength(&steps_none);
        let (_, strength_low) = compute_path_strength(&steps_low);
        assert!(
            (strength_none - strength_low).abs() < 0.001,
            "None tension ({}) and low tension ({}) should be identical",
            strength_none,
            strength_low
        );
    }

    #[tokio::test]
    async fn test_influence_tension_weighted_paths() {
        // Alice -> Bob (high tension), Alice -> Charlie (calm)
        let mut edges = std::collections::HashMap::new();
        edges.insert(
            "alice".to_string(),
            vec![
                make_edge_with_tension("bob", "rivalry", 9),
                make_edge("charlie", "mentor"),
            ],
        );

        let mut names = std::collections::HashMap::new();
        names.insert("alice".to_string(), "Alice".to_string());
        names.insert("bob".to_string(), "Bob".to_string());
        names.insert("charlie".to_string(), "Charlie".to_string());

        let provider = MockInfluenceDataProvider {
            edges,
            names,
            all_ids: vec![
                "character:alice".to_string(),
                "character:bob".to_string(),
                "character:charlie".to_string(),
            ],
        };
        let service = InfluenceService::with_provider(Arc::new(provider));

        let result = service
            .trace_propagation("character:alice", 3)
            .await
            .unwrap();

        // Both should be reachable
        assert_eq!(result.reachable_characters.len(), 2);

        // Charlie (mentor, no tension) should have higher numeric_strength than Bob (rivalry, high tension)
        let bob_path = result
            .reachable_characters
            .iter()
            .find(|p| p.steps.last().map(|s| s.character_name.as_str()) == Some("Bob"))
            .unwrap();
        let charlie_path = result
            .reachable_characters
            .iter()
            .find(|p| p.steps.last().map(|s| s.character_name.as_str()) == Some("Charlie"))
            .unwrap();

        assert!(
            charlie_path.numeric_strength > bob_path.numeric_strength,
            "Mentor path ({}) should be stronger than rivalry+tension path ({})",
            charlie_path.numeric_strength,
            bob_path.numeric_strength
        );
    }
}
