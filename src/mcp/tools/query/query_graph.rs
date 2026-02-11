use crate::mcp::NarraServer;
use crate::mcp::{EntityResult, GraphFormat, QueryResponse};
use crate::repository::RelationshipRepository;

impl NarraServer {
    pub(crate) async fn handle_graph_traversal(
        &self,
        entity_id: &str,
        depth: usize,
        _format: GraphFormat,
    ) -> Result<QueryResponse, String> {
        // Use relationship repository for graph traversal
        let connected = self
            .relationship_repo
            .get_connected_entities(entity_id, depth)
            .await
            .map_err(|e| format!("Graph traversal failed: {}", e))?;

        let results: Vec<EntityResult> = connected
            .iter()
            .map(|id| EntityResult {
                id: id.clone(),
                entity_type: self.detect_entity_type(id),
                name: self.extract_name_from_id(id),
                content: String::new(), // Minimal content for traversal
                confidence: None,
                last_modified: None,
            })
            .collect();

        let hints = vec![
            format!(
                "Found {} connected entities within {} hops",
                results.len(),
                depth
            ),
            "Use lookup on specific entities for full details".to_string(),
        ];

        Ok(QueryResponse {
            results,
            total: connected.len(),
            next_cursor: None,
            hints,
            token_estimate: connected.len() * 20, // Minimal estimate for IDs only
            truncated: None,
        })
    }

    pub(crate) async fn handle_connection_path(
        &self,
        from_id: &str,
        to_id: &str,
        max_hops: usize,
        include_events: bool,
    ) -> Result<QueryResponse, String> {
        use crate::services::graph::MermaidGraphService;

        // Create graph service instance
        let graph_service = MermaidGraphService::new(self.db.clone());

        // Call find_connection_paths
        let paths = graph_service
            .find_connection_paths(from_id, to_id, max_hops, include_events)
            .await
            .map_err(|e| format!("Connection path search failed: {}", e))?;

        if paths.is_empty() {
            let content = format!(
                "No connection found between {} and {} within {} hops",
                from_id, to_id, max_hops
            );
            let result = EntityResult {
                id: "no-connection".to_string(),
                entity_type: "path".to_string(),
                name: "No Connection".to_string(),
                content,
                confidence: Some(0.0),
                last_modified: None,
            };

            return Ok(QueryResponse {
                results: vec![result],
                total: 0,
                next_cursor: None,
                hints: vec![
                    format!("No paths found within {} hops", max_hops),
                    "Try increasing max_hops or check if both characters exist".to_string(),
                ],
                token_estimate: 100,
                truncated: None,
            });
        }

        // Format paths as EntityResults
        let mut entity_results = Vec::new();
        for (idx, path) in paths.iter().enumerate() {
            let mut path_description = format!("Path {} ({} hops):\n", idx + 1, path.total_hops);
            for (i, step) in path.steps.iter().enumerate() {
                if i == 0 {
                    path_description.push_str(&format!("  Start: {}\n", step.entity_id));
                } else {
                    path_description
                        .push_str(&format!("  {} {}\n", step.connection_type, step.entity_id));
                }
            }

            entity_results.push(EntityResult {
                id: format!("path-{}", idx),
                entity_type: "path".to_string(),
                name: format!("Path {} ({} hops)", idx + 1, path.total_hops),
                content: path_description,
                confidence: Some(1.0 / (path.total_hops as f32 + 1.0)),
                last_modified: None,
            });
        }

        let hints = vec![
            format!(
                "Found {} connection path(s) between {} and {}",
                paths.len(),
                from_id,
                to_id
            ),
            format!(
                "Shortest path: {} hops",
                paths.first().map(|p| p.total_hops).unwrap_or(0)
            ),
            "Paths are sorted by length (shortest first)".to_string(),
        ];

        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        Ok(QueryResponse {
            results: entity_results,
            total: paths.len(),
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_centrality_metrics(
        &self,
        scope: Option<String>,
        metrics: Option<Vec<String>>,
        limit: usize,
    ) -> Result<QueryResponse, String> {
        use crate::services::{CentralityMetric, GraphAnalyticsService};

        // Create GraphAnalyticsService
        let graph_service = GraphAnalyticsService::new(self.db.clone());

        // Parse metrics strings to CentralityMetric enum (default to All)
        let metric_list = if let Some(m) = metrics {
            m.iter()
                .filter_map(|s| match s.to_lowercase().as_str() {
                    "degree" => Some(CentralityMetric::Degree),
                    "betweenness" => Some(CentralityMetric::Betweenness),
                    "closeness" => Some(CentralityMetric::Closeness),
                    "all" => Some(CentralityMetric::All),
                    _ => None,
                })
                .collect()
        } else {
            vec![CentralityMetric::All]
        };

        // Call compute_centrality
        let centrality_results = graph_service
            .compute_centrality(scope, metric_list, limit)
            .await
            .map_err(|e| format!("Centrality computation failed: {}", e))?;

        // Convert CentralityResult items to EntityResult
        let entity_results: Vec<EntityResult> = centrality_results
            .iter()
            .map(|c| {
                let content = format!(
                    "Degree: {:.2} | Betweenness: {:.2} | Closeness: {:.2} | Role: {}",
                    c.degree, c.betweenness, c.closeness, c.narrative_role
                );

                EntityResult {
                    id: c.character_id.clone(),
                    entity_type: "character".to_string(),
                    name: c.character_name.clone(),
                    content,
                    confidence: Some(c.degree as f32),
                    last_modified: None,
                }
            })
            .collect();

        let hints = vec![
            "Centrality reveals structural importance in the character network".to_string(),
            "High betweenness = bridge between groups".to_string(),
            "Use scope parameter to focus on a character's neighborhood".to_string(),
        ];

        let token_estimate = self.estimate_tokens_from_results(&entity_results);

        Ok(QueryResponse {
            results: entity_results,
            total: centrality_results.len(),
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_influence_propagation(
        &self,
        from_character_id: &str,
        knowledge_fact: Option<String>,
        max_depth: usize,
    ) -> Result<QueryResponse, String> {
        use crate::services::InfluenceService;

        // Create InfluenceService
        let influence_service = InfluenceService::new(self.db.clone());

        // Call trace_propagation or trace_knowledge_propagation based on knowledge_fact
        let propagation_result = if let Some(ref fact) = knowledge_fact {
            influence_service
                .trace_knowledge_propagation(from_character_id, fact, max_depth)
                .await
        } else {
            influence_service
                .trace_propagation(from_character_id, max_depth)
                .await
        }
        .map_err(|e| format!("Influence propagation failed: {}", e))?;

        // Convert PropagationResult to QueryResponse
        // Each InfluencePath becomes an EntityResult
        let entity_results: Vec<EntityResult> = propagation_result
            .reachable_characters
            .iter()
            .map(|path| {
                // Build path description
                let mut path_str = String::new();
                for (i, step) in path.steps.iter().enumerate() {
                    if i == 0 {
                        path_str.push_str(&step.character_name);
                    } else {
                        path_str.push_str(&format!(
                            " -> ({}) -> {}",
                            step.relationship_type, step.character_name
                        ));
                    }
                }
                path_str.push_str(&format!(" | Strength: {}", path.path_strength));

                let endpoint_id = path
                    .steps
                    .last()
                    .map(|s| s.character_id.clone())
                    .unwrap_or_default();
                let endpoint_name = path
                    .steps
                    .last()
                    .map(|s| s.character_name.clone())
                    .unwrap_or_default();

                EntityResult {
                    id: endpoint_id,
                    entity_type: "influence_path".to_string(),
                    name: endpoint_name,
                    content: path_str,
                    confidence: Some(1.0 / (path.total_hops as f32 + 1.0)),
                    last_modified: None,
                }
            })
            .collect();

        let mut hints = vec![format!(
            "Found {} reachable characters from {}",
            entity_results.len(),
            from_character_id
        )];

        if !propagation_result.unreachable_characters.is_empty() {
            hints.push(format!(
                "{} characters unreachable within {} hops",
                propagation_result.unreachable_characters.len(),
                max_depth
            ));
        }

        if knowledge_fact.is_some() {
            hints.push("Shows who could learn this knowledge through relationships".to_string());
        }

        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        Ok(QueryResponse {
            results: entity_results,
            total: propagation_result.reachable_characters.len(),
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }
}
