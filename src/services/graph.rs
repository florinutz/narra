//! Graph visualization service for character relationships.
//!
//! Generates Mermaid diagram format for rendering in GitHub, Obsidian,
//! and other markdown-compatible editors.

use crate::db::connection::NarraDb;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use crate::models::character::get_character;
use crate::models::perception::{get_perceptions_from, get_perceptions_of};
use crate::models::{Character, Perception};
use crate::NarraError;

#[derive(Debug, Deserialize)]
struct PerceptionEdge {
    target: surrealdb::sql::Thing,
    rel_types: Vec<String>,
}

/// Scope of the graph to generate.
#[derive(Debug, Clone)]
pub enum GraphScope {
    /// Full network of all characters
    FullNetwork,
    /// Character-centered with limited depth
    CharacterCentered { character_id: String, depth: usize },
}

/// Options for graph generation.
#[derive(Debug, Clone)]
pub struct GraphOptions {
    /// Include character roles in node labels
    pub include_roles: bool,
    /// Direction of graph layout (TB, LR, etc.)
    pub direction: String,
}

impl Default for GraphOptions {
    fn default() -> Self {
        Self {
            include_roles: false,
            direction: "TB".to_string(),
        }
    }
}

/// Mermaid color styles by relationship type.
fn relationship_color(rel_type: &str) -> &'static str {
    match rel_type.to_lowercase().as_str() {
        "family" => "stroke:#22c55e,stroke-width:2px", // green
        "romantic" => "stroke:#ef4444,stroke-width:2px", // red
        "professional" => "stroke:#3b82f6,stroke-width:2px", // blue
        "friendship" => "stroke:#f59e0b,stroke-width:2px", // amber
        "rivalry" => "stroke:#8b5cf6,stroke-width:2px", // purple
        "mentorship" => "stroke:#14b8a6,stroke-width:2px", // teal
        "alliance" => "stroke:#6366f1,stroke-width:2px", // indigo
        _ => "stroke:#6b7280,stroke-width:1px",        // gray (unknown)
    }
}

/// Get CSS class name for relationship type.
fn relationship_class(rel_type: &str) -> String {
    format!("rel_{}", rel_type.to_lowercase().replace(' ', "_"))
}

/// Result of a reverse query - entities that reference the target
#[derive(Debug, Serialize)]
pub struct ReverseQueryResult {
    pub entity_type: String,
    pub entity_id: String,
    pub reference_field: String,
}

/// A single step in a connection path
#[derive(Debug, Clone, Serialize)]
pub struct ConnectionStep {
    pub entity_id: String,
    pub connection_type: String,
}

/// A complete path between two characters
#[derive(Debug, Clone, Serialize)]
pub struct ConnectionPathResult {
    pub steps: Vec<ConnectionStep>,
    pub total_hops: usize,
}

/// Service trait for graph generation.
#[async_trait]
pub trait GraphService: Send + Sync {
    /// Generate a Mermaid diagram for the given scope.
    async fn generate_mermaid(
        &self,
        scope: GraphScope,
        options: GraphOptions,
    ) -> Result<String, NarraError>;
}

/// SurrealDB-backed graph service implementation.
pub struct MermaidGraphService {
    db: Arc<NarraDb>,
}

impl MermaidGraphService {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self { db }
    }

    /// Get all characters in the database.
    async fn get_all_characters(&self) -> Result<Vec<Character>, NarraError> {
        let mut result = self.db.query("SELECT * FROM character").await?;
        let characters: Vec<Character> = result.take(0)?;
        Ok(characters)
    }

    /// Get all perceptions in the database.
    async fn get_all_perceptions(&self) -> Result<Vec<Perception>, NarraError> {
        let mut result = self.db.query("SELECT * FROM perceives").await?;
        let perceptions: Vec<Perception> = result.take(0)?;
        Ok(perceptions)
    }

    /// Build character-centered graph via BFS traversal.
    ///
    /// Traverses both `perceives` and `relates_to` edges to discover
    /// characters within the specified depth, collecting all perceptions
    /// for graph rendering.
    ///
    /// IDs in the returned HashSet are bare keys (no `character:` prefix),
    /// matching what `get_character()` expects.
    async fn get_characters_within_depth(
        &self,
        start_id: &str,
        max_depth: usize,
    ) -> Result<(HashSet<String>, Vec<Perception>), NarraError> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: Vec<(String, usize)> = vec![(start_id.to_string(), 0)];
        let mut collected_perceptions: Vec<Perception> = Vec::new();

        while let Some((current_id, depth)) = queue.pop() {
            if visited.contains(&current_id) || depth > max_depth {
                continue;
            }
            visited.insert(current_id.clone());

            if depth < max_depth {
                // Get outgoing perceptions
                let outgoing = get_perceptions_from(&self.db, &current_id).await?;
                for p in &outgoing {
                    let target = p.to_character.key().to_string();
                    if !visited.contains(&target) {
                        queue.push((target, depth + 1));
                    }
                }
                collected_perceptions.extend(outgoing);

                // Get incoming perceptions
                let incoming = get_perceptions_of(&self.db, &current_id).await?;
                for p in &incoming {
                    let source = p.from_character.key().to_string();
                    if !visited.contains(&source) {
                        queue.push((source, depth + 1));
                    }
                }
                collected_perceptions.extend(incoming);

                // Also traverse relates_to edges (symmetric relationships)
                let char_ref = surrealdb::RecordId::from(("character", current_id.as_str()));
                let mut relates_response = self
                    .db
                    .query("SELECT VALUE out FROM relates_to WHERE in = $ref;\nSELECT VALUE in FROM relates_to WHERE out = $ref")
                    .bind(("ref", char_ref))
                    .await?;
                let outgoing_rels: Vec<surrealdb::sql::Thing> =
                    relates_response.take(0).unwrap_or_default();
                let incoming_rels: Vec<surrealdb::sql::Thing> =
                    relates_response.take(1).unwrap_or_default();

                for rel_target in outgoing_rels.into_iter().chain(incoming_rels) {
                    // Extract bare key from "character:xxx"
                    let full = rel_target.to_string();
                    let key = full.strip_prefix("character:").unwrap_or(&full).to_string();
                    if !visited.contains(&key) {
                        queue.push((key, depth + 1));
                    }
                }
            }
        }

        Ok((visited, collected_perceptions))
    }

    /// Format a node label.
    fn format_node_label(character: &Character, include_roles: bool) -> String {
        if include_roles && !character.roles.is_empty() {
            format!("{}\\n({})", character.name, character.roles.join(", "))
        } else {
            character.name.clone()
        }
    }

    /// Build Mermaid diagram string.
    fn build_mermaid(
        &self,
        characters: &[Character],
        perceptions: &[Perception],
        options: &GraphOptions,
    ) -> String {
        let mut lines = vec![format!("graph {}", options.direction)];

        // Create character ID to name map
        let char_map: HashMap<String, &Character> = characters
            .iter()
            .map(|c| (c.id.key().to_string(), c))
            .collect();

        // Add nodes with labels
        for char in characters {
            let id = char.id.key().to_string();
            let label = Self::format_node_label(char, options.include_roles);
            lines.push(format!("    {}[{}]", id, label));
        }

        // Track unique edges (deduplicate bidirectional perceptions)
        let mut edge_set: HashSet<(String, String, String)> = HashSet::new();

        // Add edges from perceptions
        for p in perceptions {
            let from = p.from_character.key().to_string();
            let to = p.to_character.key().to_string();

            // Only include if both characters are in the graph
            if !char_map.contains_key(&from) || !char_map.contains_key(&to) {
                continue;
            }

            let rel_type = p.rel_types.first().map(|s| s.as_str()).unwrap_or("unknown");

            // Create edge key (sorted to deduplicate bidirectional)
            let edge_key = if from < to {
                (from.clone(), to.clone(), rel_type.to_string())
            } else {
                (to.clone(), from.clone(), rel_type.to_string())
            };

            if edge_set.contains(&edge_key) {
                continue;
            }
            edge_set.insert(edge_key);

            // Use undirected line for symmetric relationships
            // (most relationships are conceptually bidirectional)
            lines.push(format!(
                "    {} --- |{}| {}",
                from,
                edge_label_escape(rel_type),
                to
            ));
        }

        // Add style classes
        lines.push(String::new());
        lines.push("    %% Relationship type styles".to_string());
        for rel_type in [
            "family",
            "romantic",
            "professional",
            "friendship",
            "rivalry",
            "mentorship",
            "alliance",
        ] {
            let style = relationship_color(rel_type);
            let class_name = relationship_class(rel_type);
            lines.push(format!("    classDef {} {}", class_name, style));
        }

        lines.join("\n")
    }

    /// Generate legend as markdown.
    fn generate_legend() -> String {
        r#"
## Legend

| Color | Relationship Type |
|-------|-------------------|
| Green | Family |
| Red | Romantic |
| Blue | Professional |
| Amber | Friendship |
| Purple | Rivalry |
| Teal | Mentorship |
| Indigo | Alliance |
| Gray | Other |
"#
        .to_string()
    }

    /// Find entities that reference a target entity.
    pub async fn get_referencing_entities(
        &self,
        entity_id: &str,
        referencing_types: Option<Vec<String>>,
        limit: usize,
    ) -> Result<Vec<ReverseQueryResult>, NarraError> {
        let mut results = Vec::new();

        // Parse entity type from ID (format: "type:id")
        let parts: Vec<&str> = entity_id.split(':').collect();
        if parts.len() != 2 {
            return Err(NarraError::Validation(format!(
                "Invalid entity ID format: {}",
                entity_id
            )));
        }
        let entity_type = parts[0];
        let entity_key = parts[1];

        // Build filter for referencing types
        let should_include = |ref_type: &str| -> bool {
            if let Some(ref types) = referencing_types {
                types.iter().any(|t| t.to_lowercase() == ref_type)
            } else {
                true
            }
        };

        match entity_type {
            "character" => {
                // Find scenes where character participates
                if should_include("scene") {
                    let query = format!(
                        "SELECT VALUE out FROM participates_in WHERE in = character:{} LIMIT {}",
                        entity_key, limit
                    );
                    let mut result = self.db.query(&query).await?;
                    let scenes: Vec<surrealdb::sql::Thing> = result.take(0).unwrap_or_default();

                    for scene_id in scenes {
                        results.push(ReverseQueryResult {
                            entity_type: "scene".to_string(),
                            entity_id: scene_id.to_string(),
                            reference_field: "participants".to_string(),
                        });
                    }
                }

                // Find events where character is involved
                if should_include("event") && results.len() < limit {
                    let remaining = limit - results.len();
                    let query = format!(
                        "SELECT VALUE out FROM involved_in WHERE in = character:{} LIMIT {}",
                        entity_key, remaining
                    );
                    let mut result = self.db.query(&query).await?;
                    let events: Vec<surrealdb::sql::Thing> = result.take(0).unwrap_or_default();

                    for event_id in events {
                        results.push(ReverseQueryResult {
                            entity_type: "event".to_string(),
                            entity_id: event_id.to_string(),
                            reference_field: "involved_characters".to_string(),
                        });
                    }
                }

                // Find knowledge about character
                if should_include("knowledge") && results.len() < limit {
                    let remaining = limit - results.len();
                    let query = format!(
                        "SELECT VALUE id FROM knowledge WHERE character = character:{} LIMIT {}",
                        entity_key, remaining
                    );
                    let mut result = self.db.query(&query).await?;
                    let knowledge: Vec<surrealdb::sql::Thing> = result.take(0).unwrap_or_default();

                    for k_id in knowledge {
                        results.push(ReverseQueryResult {
                            entity_type: "knowledge".to_string(),
                            entity_id: k_id.to_string(),
                            reference_field: "character".to_string(),
                        });
                    }
                }
            }
            "location" => {
                // Find scenes with this location as primary
                if should_include("scene") {
                    let query = format!(
                        "SELECT VALUE id FROM scene WHERE primary_location = location:{} LIMIT {}",
                        entity_key,
                        limit / 2
                    );
                    let mut result = self.db.query(&query).await?;
                    let scenes: Vec<surrealdb::sql::Thing> = result.take(0).unwrap_or_default();

                    for scene_id in scenes {
                        results.push(ReverseQueryResult {
                            entity_type: "scene".to_string(),
                            entity_id: scene_id.to_string(),
                            reference_field: "primary_location".to_string(),
                        });
                    }

                    // Find scenes with this location as secondary
                    if results.len() < limit {
                        let remaining = limit - results.len();
                        let query = format!(
                            "SELECT VALUE id FROM scene WHERE location:{} IN secondary_locations LIMIT {}",
                            entity_key, remaining
                        );
                        let mut result = self.db.query(&query).await?;
                        let scenes: Vec<surrealdb::sql::Thing> = result.take(0).unwrap_or_default();

                        for scene_id in scenes {
                            results.push(ReverseQueryResult {
                                entity_type: "scene".to_string(),
                                entity_id: scene_id.to_string(),
                                reference_field: "secondary_locations".to_string(),
                            });
                        }
                    }
                }
            }
            "event" => {
                // Find scenes that belong to this event
                if should_include("scene") {
                    let query = format!(
                        "SELECT VALUE id FROM scene WHERE event = event:{} LIMIT {}",
                        entity_key, limit
                    );
                    let mut result = self.db.query(&query).await?;
                    let scenes: Vec<surrealdb::sql::Thing> = result.take(0).unwrap_or_default();

                    for scene_id in scenes {
                        results.push(ReverseQueryResult {
                            entity_type: "scene".to_string(),
                            entity_id: scene_id.to_string(),
                            reference_field: "event".to_string(),
                        });
                    }
                }
            }
            _ => {
                return Err(NarraError::Validation(format!(
                    "Unsupported entity type for reverse query: {}",
                    entity_type
                )));
            }
        }

        Ok(results)
    }

    /// Find connection paths between two characters using BFS.
    pub async fn find_connection_paths(
        &self,
        from_id: &str,
        to_id: &str,
        max_hops: usize,
        include_events: bool,
    ) -> Result<Vec<ConnectionPathResult>, NarraError> {
        let mut paths: Vec<ConnectionPathResult> = Vec::new();

        // Parse character IDs
        let from_key = from_id.split(':').nth(1).unwrap_or(from_id);
        let to_key = to_id.split(':').nth(1).unwrap_or(to_id);

        // BFS queue: (current_id, path_so_far, visited_in_path, depth)
        let mut queue: VecDeque<(String, Vec<ConnectionStep>, HashSet<String>, usize)> =
            VecDeque::new();
        let mut initial_visited = HashSet::new();
        initial_visited.insert(from_key.to_string());

        queue.push_back((
            from_key.to_string(),
            vec![ConnectionStep {
                entity_id: format!("character:{}", from_key),
                connection_type: "start".to_string(),
            }],
            initial_visited,
            0,
        ));

        while let Some((current_id, path, visited, depth)) = queue.pop_front() {
            // Check if we reached the target
            if current_id == to_key {
                paths.push(ConnectionPathResult {
                    steps: path.clone(),
                    total_hops: depth,
                });

                // Cap at 10 paths to prevent explosion
                if paths.len() >= 10 {
                    break;
                }
                continue;
            }

            // Stop if we've gone too deep
            if depth >= max_hops {
                continue;
            }

            // Explore outgoing perceives edges
            let query_out = format!(
                "SELECT out AS target, rel_types FROM perceives WHERE in = character:{}",
                current_id
            );
            let mut result = self.db.query(&query_out).await?;
            let outgoing: Vec<PerceptionEdge> = result.take(0).unwrap_or_default();

            for edge in outgoing {
                let target_full = edge.target.to_string();
                let target_key = target_full
                    .split(':')
                    .nth(1)
                    .unwrap_or(&target_full)
                    .to_string();

                // Skip if already in this path
                if visited.contains(&target_key) {
                    continue;
                }

                let rel_type = edge
                    .rel_types
                    .first()
                    .map(|s| s.as_str())
                    .unwrap_or("relationship");

                let mut new_path = path.clone();
                new_path.push(ConnectionStep {
                    entity_id: format!("character:{}", target_key),
                    connection_type: format!("perceives:{}", rel_type),
                });

                let mut new_visited = visited.clone();
                new_visited.insert(target_key.clone());

                queue.push_back((target_key, new_path, new_visited, depth + 1));
            }

            // Explore incoming perceives edges
            let query_in = format!(
                "SELECT in AS target, rel_types FROM perceives WHERE out = character:{}",
                current_id
            );
            let mut result = self.db.query(&query_in).await?;
            let incoming: Vec<PerceptionEdge> = result.take(0).unwrap_or_default();

            for edge in incoming {
                let target_full = edge.target.to_string();
                let target_key = target_full
                    .split(':')
                    .nth(1)
                    .unwrap_or(&target_full)
                    .to_string();

                if visited.contains(&target_key) {
                    continue;
                }

                let rel_type = edge
                    .rel_types
                    .first()
                    .map(|s| s.as_str())
                    .unwrap_or("relationship");

                let mut new_path = path.clone();
                new_path.push(ConnectionStep {
                    entity_id: format!("character:{}", target_key),
                    connection_type: format!("perceives:{}", rel_type),
                });

                let mut new_visited = visited.clone();
                new_visited.insert(target_key.clone());

                queue.push_back((target_key, new_path, new_visited, depth + 1));
            }

            // If include_events, find co-participants via events
            if include_events {
                let query_events = format!(
                    "SELECT VALUE out FROM involved_in WHERE in = character:{}",
                    current_id
                );
                let mut result = self.db.query(&query_events).await?;
                let events: Vec<surrealdb::sql::Thing> = result.take(0).unwrap_or_default();

                for event_id in events {
                    let event_full = event_id.to_string();
                    let event_key = event_full.split(':').nth(1).unwrap_or(&event_full);

                    // Get other participants in this event
                    let query_coparticipants = format!(
                        "SELECT VALUE in FROM involved_in WHERE out = event:{} AND in != character:{}",
                        event_key, current_id
                    );
                    let mut result = self.db.query(&query_coparticipants).await?;
                    let coparticipants: Vec<surrealdb::sql::Thing> =
                        result.take(0).unwrap_or_default();

                    for char_id in coparticipants {
                        let char_full = char_id.to_string();
                        let char_key = char_full
                            .split(':')
                            .nth(1)
                            .unwrap_or(&char_full)
                            .to_string();

                        if visited.contains(&char_key) {
                            continue;
                        }

                        let mut new_path = path.clone();
                        new_path.push(ConnectionStep {
                            entity_id: format!("character:{}", char_key),
                            connection_type: format!("co-participant:event:{}", event_key),
                        });

                        let mut new_visited = visited.clone();
                        new_visited.insert(char_key.clone());

                        queue.push_back((char_key, new_path, new_visited, depth + 1));
                    }
                }
            }
        }

        // Sort paths by length (shortest first)
        paths.sort_by_key(|p| p.total_hops);

        Ok(paths)
    }
}

/// Escape special characters in edge labels for Mermaid.
fn edge_label_escape(label: &str) -> String {
    // Mermaid edge labels need certain chars escaped
    label
        .replace('|', "\\|")
        .replace('[', "\\[")
        .replace(']', "\\]")
}

#[async_trait]
impl GraphService for MermaidGraphService {
    async fn generate_mermaid(
        &self,
        scope: GraphScope,
        options: GraphOptions,
    ) -> Result<String, NarraError> {
        let (characters, perceptions) = match scope {
            GraphScope::FullNetwork => {
                let chars = self.get_all_characters().await?;
                let percs = self.get_all_perceptions().await?;
                (chars, percs)
            }
            GraphScope::CharacterCentered {
                character_id,
                depth,
            } => {
                let (char_ids, percs) = self
                    .get_characters_within_depth(&character_id, depth)
                    .await?;

                // Fetch character details for all IDs
                let mut chars = Vec::new();
                for id in &char_ids {
                    if let Some(c) = get_character(&self.db, id).await? {
                        chars.push(c);
                    }
                }
                (chars, percs)
            }
        };

        let mermaid = self.build_mermaid(&characters, &perceptions, &options);
        let legend = Self::generate_legend();

        Ok(format!("```mermaid\n{}\n```\n{}", mermaid, legend))
    }
}
