//! MCP query handlers for composite intelligence operations.

use std::sync::Arc;

use crate::mcp::{EntityResult, NarraServer, QueryResponse};
use crate::services::progress::ProgressReporter;
use crate::services::CompositeIntelligenceService;
use serde::Deserialize;

impl NarraServer {
    /// Fetch composite_text for an entity, falling back to name/title/description.
    async fn fetch_composite_text(&self, entity_id: &str) -> Option<String> {
        #[derive(Deserialize)]
        struct TextRow {
            composite_text: Option<String>,
            name: Option<String>,
            title: Option<String>,
            description: Option<String>,
        }

        let mut resp = self
            .db
            .query(format!(
                "SELECT composite_text, name, title, description FROM {} LIMIT 1",
                entity_id
            ))
            .await
            .ok()?;

        let entity: Option<TextRow> = resp.take(0).ok().flatten();
        entity.and_then(|e| e.composite_text.or(e.description).or(e.name).or(e.title))
    }

    pub(crate) async fn handle_situation_report(
        &self,
        detail_level: Option<String>,
        token_budget: usize,
        progress: Arc<dyn ProgressReporter>,
    ) -> Result<QueryResponse, String> {
        progress.step(1, 3, "Gathering world intelligence").await;
        let service = CompositeIntelligenceService::new(self.db.clone());
        let report = service
            .situation_report()
            .await
            .map_err(|e| format!("Situation report failed: {}", e))?;
        progress.step(2, 3, "Formatting situation report").await;

        // Auto-select detail level from token budget when not explicitly specified
        let effective_detail = detail_level.as_deref().unwrap_or(match token_budget {
            0..=1500 => "summary",
            1501..=3500 => "standard",
            _ => "full",
        });
        let is_full = effective_detail == "full";
        let is_summary = effective_detail == "summary";
        let max_items = if is_summary {
            5
        } else if is_full {
            usize::MAX
        } else {
            10
        };

        let mut content_parts = Vec::new();

        // Irony highlights
        if !report.irony_highlights.is_empty() {
            let shown = report.irony_highlights.len().min(max_items);
            content_parts.push(format!(
                "## Dramatic Irony ({} highlights)",
                report.irony_highlights.len()
            ));
            for a in report.irony_highlights.iter().take(shown) {
                content_parts.push(format!(
                    "- {} knows \"{}\" but {} doesn't (weight: {:.1}, about: {})",
                    a.knowing_character_name,
                    a.fact,
                    a.unknowing_character_name,
                    a.dramatic_weight,
                    a.about
                ));
            }
            if shown < report.irony_highlights.len() {
                content_parts.push(format!(
                    "\n_{} more highlights not shown. Use detail_level='full' for complete report._",
                    report.irony_highlights.len() - shown
                ));
            }
        }

        // Knowledge conflicts
        if !report.knowledge_conflicts.is_empty() {
            let shown = report.knowledge_conflicts.len().min(max_items);
            content_parts.push(format!(
                "\n## Knowledge Conflicts ({} BelievesWrongly)",
                report.knowledge_conflicts.len()
            ));
            for c in report.knowledge_conflicts.iter().take(shown) {
                content_parts.push(format!(
                    "- {} believes wrongly about {} (truth: {})",
                    c.character_id, c.target, c.truth_value
                ));
            }
            if shown < report.knowledge_conflicts.len() {
                content_parts.push(format!(
                    "\n_{} more conflicts not shown. Use detail_level='full' for complete report._",
                    report.knowledge_conflicts.len() - shown
                ));
            }
        }

        // Tensions
        if !report.high_tension_pairs.is_empty() {
            let shown = report.high_tension_pairs.len().min(max_items);
            content_parts.push(format!(
                "\n## High Tension Pairs ({})",
                report.high_tension_pairs.len()
            ));
            for t in report.high_tension_pairs.iter().take(shown) {
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
            if shown < report.high_tension_pairs.len() {
                content_parts.push(format!(
                    "\n_{} more pairs not shown. Use detail_level='full' for complete report._",
                    report.high_tension_pairs.len() - shown
                ));
            }
        }

        // Narrative tensions (structural)
        if !report.narrative_tensions.is_empty() {
            let shown = report.narrative_tensions.len().min(max_items);
            content_parts.push(format!(
                "\n## Narrative Tensions ({})",
                report.narrative_tensions.len()
            ));
            for t in report.narrative_tensions.iter().take(shown) {
                content_parts.push(format!(
                    "- **{}** between {} and {} (severity: {:.2}): {}",
                    t.tension_type,
                    t.character_a_name,
                    t.character_b_name,
                    t.severity,
                    t.description
                ));
            }
            if shown < report.narrative_tensions.len() {
                content_parts.push(format!(
                    "\n_{} more tensions not shown. Use detail_level='full' for complete report._",
                    report.narrative_tensions.len() - shown
                ));
            }
        }

        // Themes
        content_parts.push(format!("\n## Themes: {} clusters", report.theme_count));

        // Narrative momentum
        content_parts.push("\n## Narrative Momentum".to_string());
        let momentum_label = match &report.narrative_momentum {
            crate::services::NarrativeMomentum::Accelerating { reason } => {
                format!("**Accelerating** — {}", reason)
            }
            crate::services::NarrativeMomentum::Stalling { reason } => {
                format!("**Stalling** — {}", reason)
            }
            crate::services::NarrativeMomentum::Climactic { reason } => {
                format!("**Climactic** — {}", reason)
            }
        };
        content_parts.push(momentum_label);

        // Unresolved threads
        if !report.unresolved_threads.is_empty() {
            content_parts.push(format!(
                "\n## Unresolved Threads ({})",
                report.unresolved_threads.len()
            ));
            for thread in &report.unresolved_threads {
                let age_tag = thread
                    .age_estimate
                    .as_ref()
                    .map(|a| format!(" [{}]", a))
                    .unwrap_or_default();
                content_parts.push(format!(
                    "- **{}**: {}{}",
                    thread.thread_type, thread.description, age_tag
                ));
            }
        }

        // Character arc summaries
        if !report.character_arc_summaries.is_empty() {
            content_parts.push(format!(
                "\n## Character Arc Activity (top {})",
                report.character_arc_summaries.len()
            ));
            for arc in &report.character_arc_summaries {
                content_parts.push(format!(
                    "- {}: {} ({} snapshots, recent drift: {:.2})",
                    arc.character_name,
                    arc.arc_status,
                    arc.snapshot_count,
                    arc.recent_drift.unwrap_or(0.0)
                ));
            }
        }

        // Suggestions
        if !report.suggestions.is_empty() {
            content_parts.push("\n## Narrative Suggestions".to_string());
            for s in &report.suggestions {
                content_parts.push(format!("- {}", s));
            }
        }

        let content = content_parts.join("\n");
        let token_estimate = content.len() / 4 + 50;
        progress.step(3, 3, "Situation report complete").await;

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
            truncated: None,
        })
    }

    pub(crate) async fn handle_character_dossier(
        &self,
        character_id: &str,
        detail_level: Option<String>,
        token_budget: usize,
        progress: Arc<dyn ProgressReporter>,
    ) -> Result<QueryResponse, String> {
        progress
            .step(1, 5, "Building character intelligence profile")
            .await;
        let service = CompositeIntelligenceService::new(self.db.clone());
        let mut dossier = service
            .character_dossier(character_id)
            .await
            .map_err(|e| format!("Character dossier failed: {}", e))?;

        progress.step(2, 5, "Running ML annotations").await;
        // Enrich with ML annotations (best-effort)
        if let Some(text) = self.fetch_composite_text(character_id).await {
            if self.emotion_service.is_available() {
                dossier.emotion_profile = self
                    .emotion_service
                    .get_emotions(character_id, &text)
                    .await
                    .ok();
            }
            if self.theme_service.is_available() {
                dossier.theme_tags = self
                    .theme_service
                    .get_themes(character_id, &text, None)
                    .await
                    .ok();
            }
        }
        progress.step(3, 5, "Formatting dossier").await;

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

        // Inferred roles
        if let Some(inferred) = &dossier.inferred_roles {
            content_parts.push(format!(
                "Inferred role: **{}** ({}% confidence)",
                inferred.primary_role,
                (inferred.confidence * 100.0) as i32
            ));
            if !inferred.secondary_roles.is_empty() {
                content_parts.push(format!(
                    "Secondary roles: {}",
                    inferred.secondary_roles.join(", ")
                ));
            }
        }

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

        // Arc trajectory
        if let Some(arc) = &dossier.arc_trajectory {
            content_parts.push("\n## Arc Trajectory".to_string());
            content_parts.push(format!(
                "- Direction: {} (total drift: {:.2})",
                arc.direction, arc.total_drift
            ));
            content_parts.push(format!("- Snapshots: {}", arc.snapshot_count));
            if let Some(event) = &arc.most_recent_event {
                content_parts.push(format!("- Most recent event: {}", event));
            }
        }

        // Relationship map (with detail level filtering)
        if !dossier.relationship_map.is_empty() {
            // Auto-select detail level from token budget when not explicitly specified
            let effective_detail = detail_level.as_deref().unwrap_or(match token_budget {
                0..=1500 => "summary",
                1501..=3500 => "standard",
                _ => "full",
            });
            let is_full = effective_detail == "full";
            let is_summary = effective_detail == "summary";
            let max_rels = if is_summary {
                3
            } else if is_full {
                usize::MAX
            } else {
                10
            };

            let shown = dossier.relationship_map.len().min(max_rels);
            content_parts.push(format!(
                "\n## Relationships ({})",
                dossier.relationship_map.len()
            ));
            for rel in dossier.relationship_map.iter().take(shown) {
                let tension_str = rel
                    .tension
                    .map(|t| format!(", tension: {}", t))
                    .unwrap_or_default();
                content_parts.push(format!(
                    "- {} ({}){}",
                    rel.other_name, rel.rel_type, tension_str
                ));
            }
            if shown < dossier.relationship_map.len() {
                content_parts.push(format!(
                    "\n_{} more relationships not shown. Use detail_level='full' for complete list._",
                    dossier.relationship_map.len() - shown
                ));
            }
        }

        // Knowledge inventory
        content_parts.push("\n## Knowledge Inventory".to_string());
        let inv = &dossier.knowledge_inventory;
        content_parts.push(format!(
            "- Knows: {} | Suspects: {} | BelievesWrongly: {} | Uncertain: {} | Other: {}",
            inv.knows, inv.suspects, inv.believes_wrongly, inv.uncertain, inv.other
        ));
        let total = inv.knows + inv.suspects + inv.believes_wrongly + inv.uncertain + inv.other;
        content_parts.push(format!("- Total knowledge states: {}", total));

        // Narrative tensions
        if !dossier.narrative_tensions.is_empty() {
            content_parts.push(format!(
                "\n## Narrative Tensions ({})",
                dossier.narrative_tensions.len()
            ));
            for t in &dossier.narrative_tensions {
                content_parts.push(format!(
                    "- **{} ↔ {}** [{}] severity {:.0}%: {}",
                    t.character_a_name,
                    t.character_b_name,
                    t.tension_type,
                    t.severity * 100.0,
                    t.description
                ));
            }
        }

        // Emotion profile (ML)
        if let Some(ref emotions) = dossier.emotion_profile {
            content_parts.push(format!(
                "\n## Emotion Profile (dominant: {})",
                emotions.dominant
            ));
            for s in emotions.scores.iter().take(emotions.active_count.max(3)) {
                content_parts.push(format!("- {}: {:.3}", s.label, s.score));
            }
        }

        // Theme tags (ML)
        if let Some(ref themes) = dossier.theme_tags {
            content_parts.push(format!("\n## Theme Tags (dominant: {})", themes.dominant));
            for s in themes.themes.iter().take(themes.active_count.max(3)) {
                content_parts.push(format!("- {}: {:.3}", s.label, s.score));
            }
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
        progress.step(5, 5, "Dossier complete").await;

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
            truncated: None,
        })
    }

    pub(crate) async fn handle_scene_planning(
        &self,
        character_ids: &[String],
        detail_level: Option<String>,
        token_budget: usize,
        progress: Arc<dyn ProgressReporter>,
    ) -> Result<QueryResponse, String> {
        if character_ids.len() < 2 {
            return Err("Scene planning requires at least 2 characters".to_string());
        }

        progress.step(1, 4, "Analyzing character dynamics").await;
        let service = CompositeIntelligenceService::new(self.db.clone());
        let mut plan = service
            .scene_prep(character_ids)
            .await
            .map_err(|e| format!("Scene planning failed: {}", e))?;

        progress.step(2, 4, "Running theme analysis").await;
        // Enrich with ML theme analysis (combine character texts, best-effort)
        if self.theme_service.is_available() {
            let mut combined_text = String::new();
            for char_id in &plan.characters {
                if let Some(text) = self.fetch_composite_text(char_id).await {
                    if !combined_text.is_empty() {
                        combined_text.push_str(". ");
                    }
                    combined_text.push_str(&text);
                }
            }
            if !combined_text.is_empty() {
                plan.scene_themes = self
                    .theme_service
                    .classify_themes(&combined_text, None)
                    .await
                    .ok();
            }
        }
        progress.step(3, 4, "Formatting scene plan").await;

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

        // Auto-select detail level from token budget when not explicitly specified
        let effective_detail = detail_level.as_deref().unwrap_or(match token_budget {
            0..=1500 => "summary",
            1501..=3500 => "standard",
            _ => "full",
        });
        let is_full = effective_detail == "full";
        let is_summary = effective_detail == "summary";
        let max_dynamics = if is_summary {
            5
        } else if is_full {
            usize::MAX
        } else {
            10
        };
        let max_facts = if is_summary {
            5
        } else if is_full {
            usize::MAX
        } else {
            10
        };

        // Pair dynamics
        content_parts.push("\n## Pair Dynamics".to_string());
        let shown_dynamics = plan.pair_dynamics.len().min(max_dynamics);
        for d in plan.pair_dynamics.iter().take(shown_dynamics) {
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
        if shown_dynamics < plan.pair_dynamics.len() {
            content_parts.push(format!(
                "\n_{} more dynamics not shown. Use detail_level='full' for complete list._",
                plan.pair_dynamics.len() - shown_dynamics
            ));
        }

        // Facts
        if !plan.applicable_facts.is_empty() {
            let shown_facts = plan.applicable_facts.len().min(max_facts);
            content_parts.push("\n## Applicable Universe Facts".to_string());
            for f in plan.applicable_facts.iter().take(shown_facts) {
                content_parts.push(format!("- {}", f));
            }
            if shown_facts < plan.applicable_facts.len() {
                content_parts.push(format!(
                    "\n_{} more facts not shown. Use detail_level='full' for complete list._",
                    plan.applicable_facts.len() - shown_facts
                ));
            }
        }

        // Opportunities
        if !plan.opportunities.is_empty() {
            content_parts.push("\n## Opportunities".to_string());
            for o in &plan.opportunities {
                content_parts.push(format!("- {}", o));
            }
        }

        // Knowledge reveals
        if !plan.knowledge_reveals.is_empty() {
            content_parts.push(format!(
                "\n## Knowledge Reveal Opportunities ({})",
                plan.knowledge_reveals.len()
            ));
            for reveal in &plan.knowledge_reveals {
                content_parts.push(format!(
                    "- {} could reveal \"{}\" to {} (potential: {:.1})",
                    reveal.revealer, reveal.fact, reveal.learner, reveal.dramatic_potential
                ));
            }
        }

        // Narrative tensions among scene characters
        if !plan.narrative_tensions.is_empty() {
            content_parts.push(format!(
                "\n## Narrative Tensions ({})",
                plan.narrative_tensions.len()
            ));
            for t in &plan.narrative_tensions {
                content_parts.push(format!(
                    "- **{} ↔ {}** [{}] severity {:.0}%: {}",
                    t.character_a_name,
                    t.character_b_name,
                    t.tension_type,
                    t.severity * 100.0,
                    t.description
                ));
            }
        }

        // Character roles in this scene
        if !plan.character_roles.is_empty() {
            content_parts.push("\n## Character Roles".to_string());
            for r in &plan.character_roles {
                let secondary = if r.secondary_roles.is_empty() {
                    String::new()
                } else {
                    format!(" (also: {})", r.secondary_roles.join(", "))
                };
                content_parts.push(format!(
                    "- **{}**: {} ({}% confidence){}",
                    r.character_name,
                    r.primary_role,
                    (r.confidence * 100.0) as i32,
                    secondary
                ));
            }
        }

        // Scene themes (ML)
        if let Some(ref themes) = plan.scene_themes {
            content_parts.push(format!("\n## Scene Themes (dominant: {})", themes.dominant));
            for s in themes.themes.iter().take(themes.active_count.max(3)) {
                content_parts.push(format!("- {}: {:.3}", s.label, s.score));
            }
        }

        // Fact constraints
        if !plan.fact_constraints.is_empty() {
            content_parts.push(format!(
                "\n## Universe Fact Constraints ({})",
                plan.fact_constraints.len()
            ));
            for constraint in &plan.fact_constraints {
                content_parts.push(format!(
                    "- [{}] {}: {}",
                    constraint.enforcement_level, constraint.title, constraint.relevance
                ));
            }
        }

        let content = content_parts.join("\n");
        let token_estimate = content.len() / 4 + 50;
        progress.step(4, 4, "Scene plan complete").await;

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
            truncated: None,
        })
    }
}
