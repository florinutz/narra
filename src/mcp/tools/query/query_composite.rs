//! MCP query handlers for composite intelligence operations.

use crate::mcp::{EntityResult, NarraServer, QueryResponse};
use crate::services::CompositeIntelligenceService;

impl NarraServer {
    pub(crate) async fn handle_situation_report(&self) -> Result<QueryResponse, String> {
        let service = CompositeIntelligenceService::new(self.db.clone());
        let report = service
            .situation_report()
            .await
            .map_err(|e| format!("Situation report failed: {}", e))?;

        let mut content_parts = Vec::new();

        // Irony highlights
        if !report.irony_highlights.is_empty() {
            content_parts.push(format!(
                "## Dramatic Irony ({} highlights)",
                report.irony_highlights.len()
            ));
            for a in &report.irony_highlights {
                content_parts.push(format!(
                    "- {} knows \"{}\" but {} doesn't (weight: {:.1}, about: {})",
                    a.knowing_character_name,
                    a.fact,
                    a.unknowing_character_name,
                    a.dramatic_weight,
                    a.about
                ));
            }
        }

        // Knowledge conflicts
        if !report.knowledge_conflicts.is_empty() {
            content_parts.push(format!(
                "\n## Knowledge Conflicts ({} BelievesWrongly)",
                report.knowledge_conflicts.len()
            ));
            for c in &report.knowledge_conflicts {
                content_parts.push(format!(
                    "- {} believes wrongly about {} (truth: {})",
                    c.character_id, c.target, c.truth_value
                ));
            }
        }

        // Tensions
        if !report.high_tension_pairs.is_empty() {
            content_parts.push(format!(
                "\n## High Tension Pairs ({})",
                report.high_tension_pairs.len()
            ));
            for t in &report.high_tension_pairs {
                content_parts.push(format!(
                    "- {} → {} (tension: {}{})",
                    t.observer,
                    t.target,
                    t.tension_level,
                    t.feelings
                        .as_ref()
                        .map(|f| format!(", feels: {}", f))
                        .unwrap_or_default()
                ));
            }
        }

        // Themes
        content_parts.push(format!("\n## Themes: {} clusters", report.theme_count));

        // Suggestions
        if !report.suggestions.is_empty() {
            content_parts.push("\n## Narrative Suggestions".to_string());
            for s in &report.suggestions {
                content_parts.push(format!("- {}", s));
            }
        }

        let content = content_parts.join("\n");
        let token_estimate = content.len() / 4 + 50;

        Ok(QueryResponse {
            results: vec![EntityResult {
                id: "report:situation".to_string(),
                entity_type: "report".to_string(),
                name: "Narrative Situation Report".to_string(),
                content,
                confidence: None,
                last_modified: None,
            }],
            total: 1,
            next_cursor: None,
            hints: vec![
                "Use character_dossier for deep analysis of a specific character".to_string(),
                "Use scene_planning to prepare a scene between characters".to_string(),
            ],
            token_estimate,
        })
    }

    pub(crate) async fn handle_character_dossier(
        &self,
        character_id: &str,
    ) -> Result<QueryResponse, String> {
        let service = CompositeIntelligenceService::new(self.db.clone());
        let dossier = service
            .character_dossier(character_id)
            .await
            .map_err(|e| format!("Character dossier failed: {}", e))?;

        let mut content_parts = vec![
            format!("# Dossier: {}", dossier.name),
            format!(
                "Roles: {}",
                if dossier.roles.is_empty() {
                    "none".to_string()
                } else {
                    dossier.roles.join(", ")
                }
            ),
        ];

        // Network position
        content_parts.push("\n## Network Position".to_string());
        match dossier.centrality_rank {
            Some(rank) => content_parts.push(format!("- Centrality rank: #{}", rank)),
            None => content_parts.push("- Centrality rank: unranked".to_string()),
        }
        content_parts.push(format!(
            "- Influence reach: {} characters",
            dossier.influence_reach
        ));

        // Knowledge
        content_parts.push("\n## Knowledge".to_string());
        content_parts.push(format!(
            "- Advantages (knows what others don't): {}",
            dossier.knowledge_advantages
        ));
        content_parts.push(format!(
            "- Blind spots (others know, they don't): {}",
            dossier.knowledge_blind_spots
        ));
        content_parts.push(format!("- False beliefs: {}", dossier.false_beliefs));

        // Perceptions
        content_parts.push("\n## How Others See Them".to_string());
        match dossier.avg_tension_toward_them {
            Some(avg) => content_parts.push(format!("- Average tension: {:.1}", avg)),
            None => content_parts.push("- No perception data".to_string()),
        }
        for p in &dossier.key_perceptions {
            content_parts.push(format!(
                "- {} — tension: {}, feels: {}",
                p.observer,
                p.tension_level
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                p.feelings.as_deref().unwrap_or("-")
            ));
        }

        // Suggestions
        if !dossier.suggestions.is_empty() {
            content_parts.push("\n## Suggestions".to_string());
            for s in &dossier.suggestions {
                content_parts.push(format!("- {}", s));
            }
        }

        let content = content_parts.join("\n");
        let token_estimate = content.len() / 4 + 50;

        Ok(QueryResponse {
            results: vec![EntityResult {
                id: format!("report:dossier:{}", character_id),
                entity_type: "report".to_string(),
                name: format!("Dossier: {}", dossier.name),
                content,
                confidence: None,
                last_modified: None,
            }],
            total: 1,
            next_cursor: None,
            hints: vec![
                "Use scene_planning to plan a scene involving this character".to_string(),
                "Use knowledge_asymmetries to explore specific pairwise dynamics".to_string(),
            ],
            token_estimate,
        })
    }

    pub(crate) async fn handle_scene_planning(
        &self,
        character_ids: &[String],
    ) -> Result<QueryResponse, String> {
        if character_ids.len() < 2 {
            return Err("Scene planning requires at least 2 characters".to_string());
        }

        let service = CompositeIntelligenceService::new(self.db.clone());
        let plan = service
            .scene_prep(character_ids)
            .await
            .map_err(|e| format!("Scene planning failed: {}", e))?;

        let mut content_parts = vec![format!(
            "# Scene Plan: {} characters",
            plan.characters.len()
        )];
        content_parts.push(format!("Characters: {}", plan.characters.join(", ")));

        // Summary stats
        content_parts.push(format!(
            "\nIrony opportunities: {} | Shared history scenes: {}",
            plan.total_irony_opportunities, plan.shared_history_scenes
        ));
        if let Some((a, b, t)) = &plan.highest_tension_pair {
            content_parts.push(format!("Highest tension: {} ↔ {} (level {})", a, b, t));
        }

        // Pair dynamics
        content_parts.push("\n## Pair Dynamics".to_string());
        for d in &plan.pair_dynamics {
            content_parts.push(format!(
                "- {} ↔ {}: {} asymmetries, tension: {}, {} shared scenes{}",
                d.character_a,
                d.character_b,
                d.asymmetries,
                d.tension_level
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                d.shared_scene_count,
                d.feelings
                    .as_ref()
                    .map(|f| format!(", feels: {}", f))
                    .unwrap_or_default()
            ));
        }

        // Facts
        if !plan.applicable_facts.is_empty() {
            content_parts.push("\n## Applicable Universe Facts".to_string());
            for f in &plan.applicable_facts {
                content_parts.push(format!("- {}", f));
            }
        }

        // Opportunities
        if !plan.opportunities.is_empty() {
            content_parts.push("\n## Opportunities".to_string());
            for o in &plan.opportunities {
                content_parts.push(format!("- {}", o));
            }
        }

        let content = content_parts.join("\n");
        let token_estimate = content.len() / 4 + 50;

        Ok(QueryResponse {
            results: vec![EntityResult {
                id: "report:scene_plan".to_string(),
                entity_type: "report".to_string(),
                name: format!("Scene Plan ({} characters)", plan.characters.len()),
                content,
                confidence: None,
                last_modified: None,
            }],
            total: 1,
            next_cursor: None,
            hints: vec![
                "Use character_dossier for deeper analysis of individual characters".to_string(),
            ],
            token_estimate,
        })
    }
}
