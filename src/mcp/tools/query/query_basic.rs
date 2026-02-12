use crate::mcp::NarraServer;
use crate::mcp::{DetailLevel, EntityResult, QueryResponse};
use crate::repository::{EntityRepository, KnowledgeRepository};
use crate::services::{EntityType, SearchFilter};

use super::{create_cursor, parse_cursor, parse_entity_types};

impl NarraServer {
    pub(crate) async fn handle_lookup(
        &self,
        entity_id: &str,
        detail_level: DetailLevel,
    ) -> Result<QueryResponse, String> {
        // Check if this is a note lookup (notes aren't in summary service)
        if entity_id.starts_with("note:") {
            return self.handle_note_lookup(entity_id).await;
        }

        // Map MCP DetailLevel to summary service DetailLevel
        use crate::services::summary::DetailLevel as ServiceDetailLevel;
        let service_detail_level = match detail_level {
            DetailLevel::Full => ServiceDetailLevel::Full,
            DetailLevel::Standard => ServiceDetailLevel::Summary,
            DetailLevel::Summary => ServiceDetailLevel::Minimal,
        };

        // Try to get entity via summary service for detail level handling
        let summary = match service_detail_level {
            ServiceDetailLevel::Full => {
                let full = self
                    .summary_service
                    .get_full_content(entity_id)
                    .await
                    .map_err(|e| format!("Lookup failed: {}", e))?;

                match full {
                    Some(f) => crate::services::summary::EntitySummary {
                        id: f.id.clone(),
                        entity_type: f.entity_type.clone(),
                        name: f.name.clone(),
                        content: f.content,
                        is_summarized: false,
                        estimated_tokens: f.estimated_tokens,
                        source_version: String::new(),
                    },
                    None => return Err(format!("Entity not found: {}", entity_id)),
                }
            }
            _ => self
                .summary_service
                .get_entity_content(entity_id, service_detail_level)
                .await
                .map_err(|e| format!("Lookup failed: {}", e))?
                .ok_or_else(|| format!("Entity not found: {}", entity_id))?,
        };

        // Get fact count for this entity
        let fact_count = {
            use crate::models::fact::get_fact_count_for_entity;
            get_fact_count_for_entity(&self.db, entity_id)
                .await
                .unwrap_or(0)
        };

        // Append fact count to content if there are related facts
        let content = if fact_count > 0 {
            format!("{}\n\nRelated facts: {}", summary.content, fact_count)
        } else {
            summary.content
        };

        let result = EntityResult {
            id: entity_id.to_string(),
            entity_type: summary.entity_type.clone(),
            name: summary.name.clone(),
            content,
            confidence: Some(1.0),
            last_modified: None,
        };

        let mut hints = self.generate_lookup_hints(entity_id, &result).await;

        // Add fact-related hint if entity has linked facts
        if fact_count > 0 {
            hints.push(format!(
                "This {} has {} related universe facts. Use ListFacts with entity_id filter to view.",
                summary.entity_type, fact_count
            ));
        }

        let token_estimate = summary.estimated_tokens;

        Ok(QueryResponse {
            results: vec![result],
            total: 1,
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_note_lookup(
        &self,
        entity_id: &str,
    ) -> Result<QueryResponse, String> {
        use crate::models::note::{get_note, get_note_attachments};

        // Extract note key from "note:xxx" format
        let note_key = entity_id.split(':').nth(1).unwrap_or(entity_id);

        let note = get_note(&self.db, note_key)
            .await
            .map_err(|e| format!("Note lookup failed: {}", e))?
            .ok_or_else(|| format!("Note not found: {}", entity_id))?;

        // Get attachments for this note
        let attachments = get_note_attachments(&self.db, note_key)
            .await
            .map_err(|e| format!("Failed to get attachments: {}", e))?;

        let attachment_list = if attachments.is_empty() {
            "No attachments".to_string()
        } else {
            let ids: Vec<String> = attachments.iter().map(|a| a.entity.to_string()).collect();
            format!("Attached to: {}", ids.join(", "))
        };

        let content = format!("{}\n\n{}", note.body, attachment_list);
        let token_estimate = content.len() / 4 + 30;

        let result = EntityResult {
            id: entity_id.to_string(),
            entity_type: "note".to_string(),
            name: note.title.clone(),
            content,
            confidence: Some(1.0),
            last_modified: Some(note.updated_at.to_string()),
        };

        let hints = vec![
            "Use list_notes to see all notes".to_string(),
            "Use attach_note/detach_note to manage attachments".to_string(),
        ];

        Ok(QueryResponse {
            results: vec![result],
            total: 1,
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_search(
        &self,
        query: &str,
        entity_types: Option<Vec<String>>,
        limit: usize,
        cursor: Option<String>,
    ) -> Result<QueryResponse, String> {
        let offset = if let Some(ref c) = cursor {
            parse_cursor(c)?.offset
        } else {
            0
        };

        // Convert entity type strings to EntityType enum
        let type_filter = parse_entity_types(entity_types);

        // Try full-text search first
        // Fetch enough results to cover offset + limit + 1 (for has_more detection)
        let filter = SearchFilter {
            entity_types: type_filter.clone(),
            limit: Some(offset + limit + 1),
            min_score: None,
            ..Default::default()
        };

        let mut results = self
            .search_service
            .search(query, filter)
            .await
            .map_err(|e| format!("Search failed: {}", e))?;

        // If no results, fall back to fuzzy search
        if results.is_empty() {
            let fuzzy_filter = SearchFilter {
                entity_types: type_filter,
                limit: Some(offset + limit + 1),
                min_score: Some(0.7),
                ..Default::default()
            };
            results = self
                .search_service
                .fuzzy_search(query, 0.7, fuzzy_filter)
                .await
                .map_err(|e| format!("Search failed: {}", e))?;
        }

        // Apply offset - skip first `offset` results
        if offset > 0 && results.len() > offset {
            results = results.into_iter().skip(offset).collect();
        } else if offset > 0 {
            // Offset exceeds results, return empty
            results = vec![];
        }

        let has_more = results.len() > limit;
        if has_more {
            results.truncate(limit);
        }

        let entity_results: Vec<EntityResult> = results
            .iter()
            .map(|r| EntityResult {
                id: r.id.clone(),
                entity_type: r.entity_type.clone(),
                name: r.name.clone(),
                content: r.name.clone(), // SearchResult doesn't have snippet, use name
                confidence: Some(r.score),
                last_modified: None,
            })
            .collect();

        let next_cursor = if has_more {
            Some(create_cursor(offset + limit))
        } else {
            None
        };

        let hints = self.generate_search_hints(query, &entity_results);
        let token_estimate = self.estimate_tokens_from_results(&entity_results);

        Ok(QueryResponse {
            results: entity_results,
            total: results.len(),
            next_cursor,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_temporal(
        &self,
        character_id: &str,
        event_id: Option<String>,
        event_name: Option<String>,
    ) -> Result<QueryResponse, String> {
        // Resolve event if name provided instead of ID
        let resolved_event_id = if let Some(name) = event_name {
            // Search for event by name
            let filter = SearchFilter {
                entity_types: vec![EntityType::Event],
                limit: Some(1),
                min_score: Some(0.8),
                ..Default::default()
            };
            let results = self
                .search_service
                .fuzzy_search(&name, 0.8, filter)
                .await
                .map_err(|e| format!("Event resolution failed: {}", e))?;
            results.first().map(|r| r.id.clone())
        } else {
            event_id
        };

        // Extract character key (repository methods expect just key, not table:key)
        let char_key = character_id.split(':').nth(1).unwrap_or(character_id);

        // Query knowledge state via knowledge repository
        let knowledge_states = if let Some(event_id) = resolved_event_id {
            // Extract event key too
            let event_key = event_id.split(':').nth(1).unwrap_or(&event_id);
            self.knowledge_repo
                .get_knowledge_at_event(char_key, event_key)
                .await
                .map_err(|e| format!("Temporal query failed: {}", e))?
        } else {
            // get_current_knowledge requires a target, but we want all knowledge
            // Use get_character_knowledge_states instead
            self.knowledge_repo
                .get_character_knowledge_states(char_key)
                .await
                .map_err(|e| format!("Temporal query failed: {}", e))?
        };

        let results: Vec<EntityResult> = knowledge_states
            .iter()
            .map(|k| EntityResult {
                id: k.id.to_string(),
                entity_type: "knowledge".to_string(),
                name: k.target.to_string(), // Use target knowledge ID as name for now
                content: format!(
                    "Certainty: {:?}, Method: {:?}",
                    k.certainty, k.learning_method
                ),
                confidence: None,
                last_modified: Some(k.learned_at.to_string()),
            })
            .collect();

        let hints = vec![
            format!("{} knows {} things", character_id, results.len()),
            "Query with event_id to see knowledge at a specific point in time".to_string(),
        ];

        Ok(QueryResponse {
            results,
            total: knowledge_states.len(),
            next_cursor: None,
            hints,
            token_estimate: knowledge_states.len() * 50,
            truncated: None,
        })
    }

    pub(crate) async fn handle_overview(
        &self,
        entity_type: &str,
        limit: usize,
        progress: std::sync::Arc<dyn crate::services::progress::ProgressReporter>,
    ) -> Result<QueryResponse, String> {
        progress
            .report(0.0, 1.0, Some(format!("Loading {} overview", entity_type)))
            .await;
        // List entities of given type (repository methods return all, we apply limit here)
        let mut entity_results: Vec<EntityResult> = match entity_type.to_lowercase().as_str() {
            "character" => {
                let chars = self
                    .entity_repo
                    .list_characters()
                    .await
                    .map_err(|e| format!("Overview failed: {}", e))?;
                chars
                    .into_iter()
                    .map(|c| EntityResult {
                        id: c.id.to_string(),
                        entity_type: "character".to_string(),
                        name: c.name.clone(),
                        content: format!("Roles: {}", c.roles.join(", ")),
                        confidence: None,
                        last_modified: Some(c.updated_at.to_string()),
                    })
                    .collect()
            }
            "location" => {
                let locs = self
                    .entity_repo
                    .list_locations()
                    .await
                    .map_err(|e| format!("Overview failed: {}", e))?;
                locs.into_iter()
                    .map(|l| EntityResult {
                        id: l.id.to_string(),
                        entity_type: "location".to_string(),
                        name: l.name.clone(),
                        content: l.description.clone().unwrap_or_default(),
                        confidence: None,
                        last_modified: Some(l.updated_at.to_string()),
                    })
                    .collect()
            }
            "event" => {
                let events = self
                    .entity_repo
                    .list_events()
                    .await
                    .map_err(|e| format!("Overview failed: {}", e))?;
                events
                    .into_iter()
                    .map(|e| EntityResult {
                        id: e.id.to_string(),
                        entity_type: "event".to_string(),
                        name: e.title.clone(),
                        content: e.description.clone().unwrap_or_default(),
                        confidence: None,
                        last_modified: Some(e.updated_at.to_string()),
                    })
                    .collect()
            }
            "scene" => {
                let scenes = self
                    .entity_repo
                    .list_scenes()
                    .await
                    .map_err(|e| format!("Overview failed: {}", e))?;
                scenes
                    .into_iter()
                    .map(|s| EntityResult {
                        id: s.id.to_string(),
                        entity_type: "scene".to_string(),
                        name: s.title.clone(),
                        content: s.summary.clone().unwrap_or_default(),
                        confidence: None,
                        last_modified: Some(s.updated_at.to_string()),
                    })
                    .collect()
            }
            _ => return Err(format!("Unknown entity type: {}", entity_type)),
        };

        // Apply limit (repository methods don't support limit parameter)
        if entity_results.len() > limit {
            entity_results.truncate(limit);
        }

        let hints = vec![
            format!("Showing {} {}s", entity_results.len(), entity_type),
            "Use search for filtered results".to_string(),
        ];

        let total = entity_results.len();
        let token_estimate = entity_results.len() * 80;
        progress
            .report(1.0, 1.0, Some("Overview complete".into()))
            .await;

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_list_notes(
        &self,
        entity_id: Option<String>,
        limit: usize,
    ) -> Result<QueryResponse, String> {
        use crate::models::note::{get_entity_notes, list_notes};

        let notes = if let Some(ref eid) = entity_id {
            // Get notes attached to a specific entity
            get_entity_notes(&self.db, eid)
                .await
                .map_err(|e| format!("Failed to list notes: {}", e))?
        } else {
            // List all notes with pagination
            list_notes(&self.db, limit, 0)
                .await
                .map_err(|e| format!("Failed to list notes: {}", e))?
        };

        let entity_results: Vec<EntityResult> = notes
            .into_iter()
            .take(limit)
            .map(|n| EntityResult {
                id: n.id.to_string(),
                entity_type: "note".to_string(),
                name: n.title.clone(),
                content: n.body.clone(),
                confidence: None,
                last_modified: Some(n.updated_at.to_string()),
            })
            .collect();

        let hints = if entity_id.is_some() {
            vec![
                format!("Found {} notes attached to entity", entity_results.len()),
                "Use lookup on note ID for full details".to_string(),
            ]
        } else {
            vec![
                format!("Showing {} notes", entity_results.len()),
                "Filter by entity_id to see notes for a specific entity".to_string(),
            ]
        };

        let total = entity_results.len();
        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_get_fact(&self, fact_id: &str) -> Result<QueryResponse, String> {
        use crate::models::fact::get_fact;

        // Extract fact key from fact_id (handle "universe_fact:xxx" format)
        let fact_key = fact_id.split(':').next_back().unwrap_or(fact_id);

        let fact = get_fact(&self.db, fact_key)
            .await
            .map_err(|e| format!("Fact lookup failed: {}", e))?
            .ok_or_else(|| format!("Fact not found: {}", fact_id))?;

        // Format categories as comma-separated list
        let categories_str = fact
            .categories
            .iter()
            .map(|c| match c {
                crate::models::fact::FactCategory::PhysicsMagic => "physics_magic".to_string(),
                crate::models::fact::FactCategory::SocialCultural => "social_cultural".to_string(),
                crate::models::fact::FactCategory::Technology => "technology".to_string(),
                crate::models::fact::FactCategory::Custom(s) => s.clone(),
            })
            .collect::<Vec<_>>()
            .join(", ");

        let content = format!(
            "**{}**\n\n{}\n\n**Categories:** {}\n**Enforcement:** {:?}",
            fact.title,
            fact.description,
            if categories_str.is_empty() {
                "none".to_string()
            } else {
                categories_str
            },
            fact.enforcement_level
        );

        let token_estimate = content.len() / 4 + 30;

        let result = EntityResult {
            id: fact.id.to_string(),
            entity_type: "universe_fact".to_string(),
            name: fact.title.clone(),
            content,
            confidence: Some(1.0),
            last_modified: Some(fact.updated_at.to_string()),
        };

        let hints = vec![
            "Use list_facts to see all facts".to_string(),
            "Link facts to entities using graph operations".to_string(),
        ];

        Ok(QueryResponse {
            results: vec![result],
            total: 1,
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_list_facts(
        &self,
        category: Option<String>,
        enforcement_level: Option<String>,
        search: Option<String>,
        entity_id: Option<String>,
        limit: usize,
        cursor: Option<String>,
    ) -> Result<QueryResponse, String> {
        use crate::models::fact::{get_entity_facts, list_facts};

        let offset = if let Some(ref c) = cursor {
            parse_cursor(c)?.offset
        } else {
            0
        };

        // Get facts - either all facts or filtered by entity
        let mut facts = if let Some(ref eid) = entity_id {
            // Get only facts linked to this entity
            get_entity_facts(&self.db, eid)
                .await
                .map_err(|e| format!("Failed to get facts for entity: {}", e))?
        } else {
            // Get all facts
            list_facts(&self.db)
                .await
                .map_err(|e| format!("Failed to list facts: {}", e))?
        };

        // Apply category filter
        if let Some(ref cat) = category {
            facts.retain(|f| {
                f.categories.iter().any(|c| {
                    let cat_str = match c {
                        crate::models::fact::FactCategory::PhysicsMagic => "physics_magic",
                        crate::models::fact::FactCategory::SocialCultural => "social_cultural",
                        crate::models::fact::FactCategory::Technology => "technology",
                        crate::models::fact::FactCategory::Custom(s) => s.as_str(),
                    };
                    cat_str == cat
                })
            });
        }

        // Apply enforcement level filter
        if let Some(ref level) = enforcement_level {
            let target_level = match level.to_lowercase().as_str() {
                "informational" => crate::models::fact::EnforcementLevel::Informational,
                "warning" => crate::models::fact::EnforcementLevel::Warning,
                "strict" => crate::models::fact::EnforcementLevel::Strict,
                _ => crate::models::fact::EnforcementLevel::Warning,
            };
            facts.retain(|f| f.enforcement_level == target_level);
        }

        // Apply search filter (case-insensitive search on title and description)
        if let Some(ref query) = search {
            let query_lower = query.to_lowercase();
            facts.retain(|f| {
                f.title.to_lowercase().contains(&query_lower)
                    || f.description.to_lowercase().contains(&query_lower)
            });
        }

        let total = facts.len();

        // Apply pagination
        if offset > 0 && facts.len() > offset {
            facts = facts.into_iter().skip(offset).collect();
        } else if offset > 0 {
            facts = vec![];
        }

        let has_more = facts.len() > limit;
        if has_more {
            facts.truncate(limit);
        }

        // Abbreviate content for list results (>5 items)
        let should_abbreviate = facts.len() > 5;

        let entity_results: Vec<EntityResult> = facts
            .iter()
            .map(|f| {
                let categories_str = f
                    .categories
                    .iter()
                    .map(|c| match c {
                        crate::models::fact::FactCategory::PhysicsMagic => {
                            "physics_magic".to_string()
                        }
                        crate::models::fact::FactCategory::SocialCultural => {
                            "social_cultural".to_string()
                        }
                        crate::models::fact::FactCategory::Technology => "technology".to_string(),
                        crate::models::fact::FactCategory::Custom(s) => s.clone(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");

                // Abbreviate description if showing many results
                let description = if should_abbreviate {
                    let desc = &f.description;
                    if let Some(first_sentence_end) = desc.find(". ") {
                        let mut abbrev = desc[..first_sentence_end + 1].to_string();
                        if abbrev.len() > 100 {
                            abbrev.truncate(100);
                            abbrev.push_str("...");
                        }
                        abbrev
                    } else {
                        let mut abbrev = desc.clone();
                        if abbrev.len() > 100 {
                            abbrev.truncate(100);
                            abbrev.push_str("...");
                        }
                        abbrev
                    }
                } else {
                    f.description.clone()
                };

                EntityResult {
                    id: f.id.to_string(),
                    entity_type: "universe_fact".to_string(),
                    name: f.title.clone(),
                    content: format!(
                        "{} | Categories: {} | Enforcement: {:?}",
                        description,
                        if categories_str.is_empty() {
                            "none".to_string()
                        } else {
                            categories_str
                        },
                        f.enforcement_level
                    ),
                    confidence: Some(1.0),
                    last_modified: Some(f.updated_at.to_string()),
                }
            })
            .collect();

        let next_cursor = if has_more {
            Some(create_cursor(offset + limit))
        } else {
            None
        };

        let mut hints = if entity_id.is_some() {
            vec![format!(
                "Showing {} of {} facts for entity",
                entity_results.len(),
                total
            )]
        } else {
            vec![format!(
                "Showing {} of {} facts",
                entity_results.len(),
                total
            )]
        };
        if has_more {
            hints.push("Use GetFact for full details".to_string());
        }
        if entity_id.is_none() && !entity_results.is_empty() {
            hints.push("Filter by entity_id to see facts for a specific entity".to_string());
        }

        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor,
            hints,
            token_estimate,
            truncated: None,
        })
    }
}
