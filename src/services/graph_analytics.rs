//! Graph analytics service for computing centrality metrics on character networks.
//!
//! Provides centrality computation (degree, betweenness, closeness) to identify
//! structural protagonists, narrative hubs, and bridging characters in the
//! character relationship network.

use async_trait::async_trait;
use graphrs::{algorithms::centrality, Edge, Graph, GraphSpecs, Node};
use std::collections::HashMap;
use std::sync::Arc;
use surrealdb::{engine::local::Db, Surreal};

use crate::NarraError;

/// Result of centrality computation for a single character.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CentralityResult {
    pub character_id: String,
    pub character_name: String,
    pub degree: f64,
    pub betweenness: f64,
    pub closeness: f64,
    pub narrative_role: String,
}

/// Centrality metric types.
#[derive(Debug, Clone, PartialEq)]
pub enum CentralityMetric {
    Degree,
    Betweenness,
    Closeness,
    All,
}

// ---------------------------------------------------------------------------
// Pure functions
// ---------------------------------------------------------------------------

/// Assign narrative role based on centrality scores.
pub(crate) fn assign_narrative_role(degree: f64, betweenness: f64, _closeness: f64) -> String {
    if degree == 0.0 {
        "isolated".to_string()
    } else if degree > 0.5 {
        "hub".to_string()
    } else if betweenness > 0.3 && degree < 0.5 {
        "bridge".to_string()
    } else if degree < 0.2 && betweenness < 0.1 {
        "peripheral".to_string()
    } else {
        "connected".to_string()
    }
}

// ---------------------------------------------------------------------------
// Data provider trait
// ---------------------------------------------------------------------------

/// A character node for graph construction.
#[derive(Debug, Clone)]
pub struct CharacterNodeInfo {
    pub id: String,
    pub name: String,
}

/// A directed perception edge for graph construction.
#[derive(Debug, Clone)]
pub struct PerceptionEdgeInfo {
    pub from_id: String,
    pub to_id: String,
}

/// Data access abstraction for the graph analytics service.
#[async_trait]
pub trait GraphDataProvider: Send + Sync {
    async fn get_all_characters(&self) -> Result<Vec<CharacterNodeInfo>, NarraError>;
    async fn get_all_perception_edges(&self) -> Result<Vec<PerceptionEdgeInfo>, NarraError>;
    /// Get all relates_to edges (family, alliance, etc.).
    /// Characters connected only via relates_to should not appear isolated.
    async fn get_all_relationship_edges(&self) -> Result<Vec<PerceptionEdgeInfo>, NarraError>;
}

/// SurrealDB implementation of GraphDataProvider.
pub struct SurrealGraphDataProvider {
    db: Arc<Surreal<Db>>,
}

impl SurrealGraphDataProvider {
    pub fn new(db: Arc<Surreal<Db>>) -> Self {
        Self { db }
    }
}

/// Internal struct for character query results.
#[derive(Debug, Clone, serde::Deserialize)]
struct CharacterNode {
    id: surrealdb::sql::Thing,
    name: String,
}

/// Internal struct for perception edge query results.
#[derive(Debug, serde::Deserialize)]
struct PerceptionEdge {
    #[serde(rename = "in")]
    from: surrealdb::sql::Thing,
    #[serde(rename = "out")]
    to: surrealdb::sql::Thing,
}

#[async_trait]
impl GraphDataProvider for SurrealGraphDataProvider {
    async fn get_all_characters(&self) -> Result<Vec<CharacterNodeInfo>, NarraError> {
        let mut result = self.db.query("SELECT id, name FROM character").await?;
        let characters: Vec<CharacterNode> = result.take(0)?;
        Ok(characters
            .into_iter()
            .map(|c| CharacterNodeInfo {
                id: c.id.to_string(),
                name: c.name,
            })
            .collect())
    }

    async fn get_all_perception_edges(&self) -> Result<Vec<PerceptionEdgeInfo>, NarraError> {
        let mut result = self.db.query("SELECT in, out FROM perceives").await?;
        let edges: Vec<PerceptionEdge> = result.take(0)?;
        Ok(edges
            .into_iter()
            .map(|e| PerceptionEdgeInfo {
                from_id: e.from.to_string(),
                to_id: e.to.to_string(),
            })
            .collect())
    }

    async fn get_all_relationship_edges(&self) -> Result<Vec<PerceptionEdgeInfo>, NarraError> {
        #[derive(Debug, serde::Deserialize)]
        struct RelationshipEdge {
            #[serde(rename = "in")]
            from: surrealdb::sql::Thing,
            #[serde(rename = "out")]
            to: surrealdb::sql::Thing,
        }

        let mut result = self.db.query("SELECT in, out FROM relates_to").await?;
        let edges: Vec<RelationshipEdge> = result.take(0)?;
        Ok(edges
            .into_iter()
            .map(|e| PerceptionEdgeInfo {
                from_id: e.from.to_string(),
                to_id: e.to.to_string(),
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// GraphAnalyticsService
// ---------------------------------------------------------------------------

/// Service for computing graph analytics on character networks.
pub struct GraphAnalyticsService {
    data: Arc<dyn GraphDataProvider>,
}

impl GraphAnalyticsService {
    pub fn new(db: Arc<Surreal<Db>>) -> Self {
        Self {
            data: Arc::new(SurrealGraphDataProvider::new(db)),
        }
    }

    pub fn with_provider(data: Arc<dyn GraphDataProvider>) -> Self {
        Self { data }
    }

    /// Compute centrality metrics for characters in the network.
    pub async fn compute_centrality(
        &self,
        scope: Option<String>,
        metrics: Vec<CentralityMetric>,
        limit: usize,
    ) -> Result<Vec<CentralityResult>, NarraError> {
        let characters = self.data.get_all_characters().await?;
        let perception_edges = self.data.get_all_perception_edges().await?;
        let relationship_edges = self.data.get_all_relationship_edges().await?;

        // Merge and deduplicate edges from both sources
        let edges = Self::merge_edges(perception_edges, relationship_edges);

        if characters.is_empty() {
            return Ok(Vec::new());
        }

        // Apply scope filtering if specified
        let filtered_characters = if let Some(ref entity_id) = scope {
            self.filter_characters_within_hops(&characters, &edges, entity_id, 3)?
        } else {
            characters
        };

        // Handle single character case
        if filtered_characters.len() == 1 {
            let char = &filtered_characters[0];
            return Ok(vec![CentralityResult {
                character_id: char.id.clone(),
                character_name: char.name.clone(),
                degree: 0.0,
                betweenness: 0.0,
                closeness: 0.0,
                narrative_role: "isolated".to_string(),
            }]);
        }

        // Build character ID set for filtering edges
        let char_id_set: std::collections::HashSet<_> =
            filtered_characters.iter().map(|c| c.id.clone()).collect();

        // Build in-memory graphrs Graph
        let graph = self.build_graph(&filtered_characters, &edges, &char_id_set)?;

        // Compute requested metrics
        let should_compute_degree =
            metrics.contains(&CentralityMetric::Degree) || metrics.contains(&CentralityMetric::All);
        let should_compute_betweenness = metrics.contains(&CentralityMetric::Betweenness)
            || metrics.contains(&CentralityMetric::All);
        let should_compute_closeness = metrics.contains(&CentralityMetric::Closeness)
            || metrics.contains(&CentralityMetric::All);

        let degree_scores = if should_compute_degree {
            self.compute_degree_centrality(&graph, &filtered_characters)?
        } else {
            HashMap::new()
        };

        let betweenness_scores = if should_compute_betweenness {
            match centrality::betweenness::betweenness_centrality(&graph, false, true) {
                Ok(scores) => scores,
                Err(e) => {
                    return Err(NarraError::Database(format!(
                        "Betweenness centrality error: {:?}",
                        e
                    )))
                }
            }
        } else {
            HashMap::new()
        };

        let closeness_scores = if should_compute_closeness {
            match centrality::closeness::closeness_centrality(&graph, false, true) {
                Ok(scores) => scores,
                Err(e) => {
                    return Err(NarraError::Database(format!(
                        "Closeness centrality error: {:?}",
                        e
                    )))
                }
            }
        } else {
            HashMap::new()
        };

        // Map results to CentralityResult
        let mut results: Vec<CentralityResult> = filtered_characters
            .iter()
            .map(|char| {
                let char_id = &char.id;
                let degree = degree_scores.get(char_id).copied().unwrap_or(0.0);
                let betweenness = betweenness_scores.get(char_id).copied().unwrap_or(0.0);
                let closeness = closeness_scores.get(char_id).copied().unwrap_or(0.0);

                let narrative_role = assign_narrative_role(degree, betweenness, closeness);

                CentralityResult {
                    character_id: char_id.clone(),
                    character_name: char.name.clone(),
                    degree,
                    betweenness,
                    closeness,
                    narrative_role,
                }
            })
            .collect();

        // Sort by primary metric
        let primary_metric = metrics.first().unwrap_or(&CentralityMetric::Degree);
        match primary_metric {
            CentralityMetric::Degree | CentralityMetric::All => {
                results.sort_by(|a, b| {
                    b.degree
                        .partial_cmp(&a.degree)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            CentralityMetric::Betweenness => {
                results.sort_by(|a, b| {
                    b.betweenness
                        .partial_cmp(&a.betweenness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            CentralityMetric::Closeness => {
                results.sort_by(|a, b| {
                    b.closeness
                        .partial_cmp(&a.closeness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        results.truncate(limit);

        Ok(results)
    }

    /// Merge perception and relationship edges, deduplicating by (from_id, to_id).
    fn merge_edges(
        perception_edges: Vec<PerceptionEdgeInfo>,
        relationship_edges: Vec<PerceptionEdgeInfo>,
    ) -> Vec<PerceptionEdgeInfo> {
        let mut seen: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        let mut merged = Vec::with_capacity(perception_edges.len() + relationship_edges.len());

        for edge in perception_edges.into_iter().chain(relationship_edges) {
            let key = (edge.from_id.clone(), edge.to_id.clone());
            if seen.insert(key) {
                merged.push(edge);
            }
        }

        merged
    }

    /// Build a directed graph from characters and edges.
    fn build_graph(
        &self,
        characters: &[CharacterNodeInfo],
        edges: &[PerceptionEdgeInfo],
        char_id_set: &std::collections::HashSet<String>,
    ) -> Result<Graph<String, ()>, NarraError> {
        let mut graph = Graph::<String, ()>::new(GraphSpecs::directed());

        for char in characters {
            let node = Node::from_name(char.id.clone());
            graph.add_node(node);
        }

        for edge in edges {
            if char_id_set.contains(&edge.from_id) && char_id_set.contains(&edge.to_id) {
                let graph_edge = Edge::new(edge.from_id.clone(), edge.to_id.clone());
                if let Err(e) = graph.add_edge(graph_edge) {
                    return Err(NarraError::Database(format!("Failed to add edge: {:?}", e)));
                }
            }
        }

        Ok(graph)
    }

    /// Compute degree centrality by iterating the edge list once — O(N+E).
    fn compute_degree_centrality(
        &self,
        graph: &Graph<String, ()>,
        characters: &[CharacterNodeInfo],
    ) -> Result<HashMap<String, f64>, NarraError> {
        let n = characters.len();

        if n <= 1 {
            return Ok(characters.iter().map(|c| (c.id.clone(), 0.0)).collect());
        }

        // Count in-degree and out-degree from edge list in one pass
        let mut degree: HashMap<&String, usize> = HashMap::new();

        for edge in graph.get_all_edges() {
            *degree.entry(&edge.u).or_default() += 1; // out-degree
            *degree.entry(&edge.v).or_default() += 1; // in-degree
        }

        let scores = characters
            .iter()
            .map(|c| {
                let total = degree.get(&c.id).copied().unwrap_or(0);
                let normalized = (total as f64) / ((n - 1) as f64);
                (c.id.clone(), normalized)
            })
            .collect();

        Ok(scores)
    }

    /// Filter characters within N hops of the target entity using BFS.
    fn filter_characters_within_hops(
        &self,
        all_characters: &[CharacterNodeInfo],
        all_edges: &[PerceptionEdgeInfo],
        entity_id: &str,
        max_hops: usize,
    ) -> Result<Vec<CharacterNodeInfo>, NarraError> {
        use std::collections::{HashSet, VecDeque};

        let entity_key = entity_id.split(':').nth(1).unwrap_or(entity_id);
        let full_entity_id = format!("character:{}", entity_key);

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        visited.insert(full_entity_id.clone());
        queue.push_back((full_entity_id, 0));

        let mut outgoing: HashMap<String, Vec<String>> = HashMap::new();
        let mut incoming: HashMap<String, Vec<String>> = HashMap::new();

        for edge in all_edges {
            outgoing
                .entry(edge.from_id.clone())
                .or_default()
                .push(edge.to_id.clone());
            incoming
                .entry(edge.to_id.clone())
                .or_default()
                .push(edge.from_id.clone());
        }

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= max_hops {
                continue;
            }

            if let Some(neighbors) = outgoing.get(&current_id) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push_back((neighbor.clone(), depth + 1));
                    }
                }
            }

            if let Some(neighbors) = incoming.get(&current_id) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push_back((neighbor.clone(), depth + 1));
                    }
                }
            }
        }

        let filtered: Vec<CharacterNodeInfo> = all_characters
            .iter()
            .filter(|c| visited.contains(&c.id))
            .cloned()
            .collect();

        Ok(filtered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Pure function tests --

    #[test]
    fn test_assign_narrative_role_isolated() {
        assert_eq!(assign_narrative_role(0.0, 0.0, 0.0), "isolated");
    }

    #[test]
    fn test_assign_narrative_role_hub() {
        assert_eq!(assign_narrative_role(0.8, 0.5, 0.6), "hub");
        assert_eq!(assign_narrative_role(0.6, 0.0, 0.0), "hub");
    }

    #[test]
    fn test_assign_narrative_role_bridge() {
        assert_eq!(assign_narrative_role(0.3, 0.5, 0.4), "bridge");
        assert_eq!(assign_narrative_role(0.4, 0.4, 0.2), "bridge");
    }

    #[test]
    fn test_assign_narrative_role_peripheral() {
        assert_eq!(assign_narrative_role(0.1, 0.05, 0.3), "peripheral");
    }

    #[test]
    fn test_assign_narrative_role_connected() {
        assert_eq!(assign_narrative_role(0.3, 0.2, 0.5), "connected");
    }

    // -- Mock-based service tests --

    struct MockGraphDataProvider {
        characters: Vec<CharacterNodeInfo>,
        edges: Vec<PerceptionEdgeInfo>,
        relationship_edges: Vec<PerceptionEdgeInfo>,
    }

    #[async_trait]
    impl GraphDataProvider for MockGraphDataProvider {
        async fn get_all_characters(&self) -> Result<Vec<CharacterNodeInfo>, NarraError> {
            Ok(self.characters.clone())
        }

        async fn get_all_perception_edges(&self) -> Result<Vec<PerceptionEdgeInfo>, NarraError> {
            Ok(self.edges.clone())
        }

        async fn get_all_relationship_edges(&self) -> Result<Vec<PerceptionEdgeInfo>, NarraError> {
            Ok(self.relationship_edges.clone())
        }
    }

    fn char_node(id: &str, name: &str) -> CharacterNodeInfo {
        CharacterNodeInfo {
            id: format!("character:{}", id),
            name: name.to_string(),
        }
    }

    fn edge(from: &str, to: &str) -> PerceptionEdgeInfo {
        PerceptionEdgeInfo {
            from_id: format!("character:{}", from),
            to_id: format!("character:{}", to),
        }
    }

    #[tokio::test]
    async fn test_graph_empty_network() {
        let provider = MockGraphDataProvider {
            characters: vec![],
            edges: vec![],
            relationship_edges: vec![],
        };
        let service = GraphAnalyticsService::with_provider(Arc::new(provider));
        let results = service
            .compute_centrality(None, vec![CentralityMetric::All], 10)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_graph_single_node() {
        let provider = MockGraphDataProvider {
            characters: vec![char_node("alice", "Alice")],
            edges: vec![],
            relationship_edges: vec![],
        };
        let service = GraphAnalyticsService::with_provider(Arc::new(provider));
        let results = service
            .compute_centrality(None, vec![CentralityMetric::All], 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].narrative_role, "isolated");
    }

    #[tokio::test]
    async fn test_graph_triangle() {
        // A -> B, B -> C, C -> A (triangle)
        let provider = MockGraphDataProvider {
            characters: vec![
                char_node("alice", "Alice"),
                char_node("bob", "Bob"),
                char_node("charlie", "Charlie"),
            ],
            edges: vec![
                edge("alice", "bob"),
                edge("bob", "charlie"),
                edge("charlie", "alice"),
            ],
            relationship_edges: vec![],
        };
        let service = GraphAnalyticsService::with_provider(Arc::new(provider));
        let results = service
            .compute_centrality(None, vec![CentralityMetric::Degree], 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 3);
        // In a complete directed triangle, all nodes should have equal degree
        let degrees: Vec<f64> = results.iter().map(|r| r.degree).collect();
        assert!(
            (degrees[0] - degrees[1]).abs() < 1e-6 && (degrees[1] - degrees[2]).abs() < 1e-6,
            "Triangle should have equal degree centrality"
        );
    }

    #[tokio::test]
    async fn test_graph_star_topology() {
        // Hub: Alice -> Bob, Alice -> Charlie, Alice -> Dave
        let provider = MockGraphDataProvider {
            characters: vec![
                char_node("alice", "Alice"),
                char_node("bob", "Bob"),
                char_node("charlie", "Charlie"),
                char_node("dave", "Dave"),
            ],
            edges: vec![
                edge("alice", "bob"),
                edge("alice", "charlie"),
                edge("alice", "dave"),
            ],
            relationship_edges: vec![],
        };
        let service = GraphAnalyticsService::with_provider(Arc::new(provider));
        let results = service
            .compute_centrality(None, vec![CentralityMetric::Degree], 10)
            .await
            .unwrap();
        // Alice has 3 outgoing edges, others have 1 incoming each
        assert_eq!(results[0].character_name, "Alice");
        assert!(results[0].degree > results[1].degree);
    }

    #[tokio::test]
    async fn test_graph_isolated_nodes() {
        let provider = MockGraphDataProvider {
            characters: vec![char_node("alice", "Alice"), char_node("bob", "Bob")],
            edges: vec![],
            relationship_edges: vec![],
        };
        let service = GraphAnalyticsService::with_provider(Arc::new(provider));
        let results = service
            .compute_centrality(None, vec![CentralityMetric::Degree], 10)
            .await
            .unwrap();
        for r in &results {
            assert_eq!(r.degree, 0.0);
            assert_eq!(r.narrative_role, "isolated");
        }
    }

    #[tokio::test]
    async fn test_graph_relates_to_edges_included() {
        // No perceives edges, but a relates_to triangle: Alice-Bob, Bob-Charlie, Charlie-Alice
        let provider = MockGraphDataProvider {
            characters: vec![
                char_node("alice", "Alice"),
                char_node("bob", "Bob"),
                char_node("charlie", "Charlie"),
            ],
            edges: vec![], // No perceives
            relationship_edges: vec![
                edge("alice", "bob"),
                edge("bob", "charlie"),
                edge("charlie", "alice"),
            ],
        };
        let service = GraphAnalyticsService::with_provider(Arc::new(provider));
        let results = service
            .compute_centrality(None, vec![CentralityMetric::Degree], 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 3);
        // All should have non-zero degree (connected via relates_to)
        for r in &results {
            assert!(
                r.degree > 0.0,
                "Character {} should NOT be isolated (connected via relates_to)",
                r.character_name
            );
            assert_ne!(r.narrative_role, "isolated");
        }
    }

    #[tokio::test]
    async fn test_graph_deduplicates_merged_edges() {
        // Same edge in both perceives and relates_to should be counted once
        let provider = MockGraphDataProvider {
            characters: vec![char_node("alice", "Alice"), char_node("bob", "Bob")],
            edges: vec![edge("alice", "bob")],
            relationship_edges: vec![edge("alice", "bob")], // duplicate
        };
        let service = GraphAnalyticsService::with_provider(Arc::new(provider));
        let results = service
            .compute_centrality(None, vec![CentralityMetric::Degree], 10)
            .await
            .unwrap();
        // Alice has 1 out-degree, Bob has 1 in-degree, each normalized by (n-1)=1
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(
                (r.degree - 1.0).abs() < 1e-6,
                "Degree should be 1.0 (deduped edge), got {} for {}",
                r.degree,
                r.character_name
            );
        }
    }
}
