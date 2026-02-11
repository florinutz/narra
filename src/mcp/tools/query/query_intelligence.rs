//! MCP query handlers for analytical intelligence operations.
//!
//! Provides: TensionMatrix, KnowledgeGapAnalysis, RelationshipStrengthMap,
//! NarrativeThreads, CharacterVoice.

use serde::Deserialize;

use crate::mcp::types::MAX_LIMIT;
use crate::mcp::{EntityResult, NarraServer, QueryResponse};

impl NarraServer {
    pub(crate) async fn handle_tension_matrix(
        &self,
        min_tension: Option<i32>,
        limit: Option<usize>,
    ) -> Result<QueryResponse, String> {
        let min_tension = min_tension.unwrap_or(1);
        let limit = limit.unwrap_or(50).min(MAX_LIMIT);

        #[derive(Deserialize)]
        struct TensionRow {
            observer: String,
            target: String,
            tension_level: i32,
            feelings: Option<String>,
        }

        let query = format!(
            "SELECT meta::id(in) AS observer, meta::id(out) AS target, \
             tension_level, feelings \
             FROM perceives WHERE tension_level >= {} \
             ORDER BY tension_level DESC LIMIT {}",
            min_tension, limit
        );

        let mut resp = self
            .db
            .query(&query)
            .await
            .map_err(|e| format!("Tension matrix query failed: {}", e))?;
        let rows: Vec<TensionRow> = resp
            .take(0)
            .map_err(|e| format!("Failed to parse tension data: {}", e))?;

        if rows.is_empty() {
            return Ok(QueryResponse {
                results: vec![EntityResult {
                    id: "report:tension_matrix".to_string(),
                    entity_type: "report".to_string(),
                    name: "Tension Matrix".to_string(),
                    content: format!("No character pairs with tension >= {} found.", min_tension),
                    confidence: None,
                    last_modified: None,
                }],
                total: 1,
                next_cursor: None,
                hints: vec!["Try lowering min_tension to see more relationships".to_string()],
                token_estimate: 50,
                truncated: None,
            });
        }

        let mut content_parts = vec![format!(
            "# Tension Matrix ({} pairs, min tension: {})",
            rows.len(),
            min_tension
        )];

        content_parts.push("\n| Observer | Target | Tension | Feelings |".to_string());
        content_parts.push("|----------|--------|---------|----------|".to_string());

        for row in &rows {
            content_parts.push(format!(
                "| {} | {} | {} | {} |",
                row.observer,
                row.target,
                row.tension_level,
                row.feelings.as_deref().unwrap_or("-")
            ));
        }

        let max_tension = rows.iter().map(|r| r.tension_level).max().unwrap_or(0);
        let avg_tension =
            rows.iter().map(|r| r.tension_level as f32).sum::<f32>() / rows.len() as f32;

        content_parts.push(format!(
            "\n**Summary:** {} pairs | Max tension: {} | Avg tension: {:.1}",
            rows.len(),
            max_tension,
            avg_tension
        ));

        let content = content_parts.join("\n");
        let token_estimate = content.len() / 4 + 50;

        Ok(QueryResponse {
            results: vec![EntityResult {
                id: "report:tension_matrix".to_string(),
                entity_type: "report".to_string(),
                name: "Tension Matrix".to_string(),
                content,
                confidence: None,
                last_modified: None,
            }],
            total: 1,
            next_cursor: None,
            hints: vec![
                "Use perception_gap for deeper analysis of specific pairs".to_string(),
                "Use conflict_detection prompt to explore escalation paths".to_string(),
            ],
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_knowledge_gap_analysis(
        &self,
        character_id: &str,
    ) -> Result<QueryResponse, String> {
        // Get character name
        #[derive(Deserialize)]
        struct NameRow {
            name: String,
        }

        let name_query = format!("SELECT name FROM {} LIMIT 1", character_id);
        let mut resp = self
            .db
            .query(&name_query)
            .await
            .map_err(|e| format!("Failed to look up character: {}", e))?;
        let name_row: Option<NameRow> = resp
            .take(0)
            .map_err(|e| format!("Failed to parse character: {}", e))?;
        let character_name = name_row
            .map(|r| r.name)
            .unwrap_or_else(|| character_id.to_string());

        // Get all knowledge states for this character
        #[derive(Deserialize)]
        struct KnowledgeRow {
            target: String,
            certainty: String,
            fact: Option<String>,
            truth_value: Option<String>,
        }

        let knowledge_query = format!(
            "SELECT meta::id(out) AS target, certainty, out.fact AS fact, \
             truth_value, learned_at \
             FROM knows WHERE in = {} ORDER BY learned_at DESC",
            character_id
        );
        let mut resp = self
            .db
            .query(&knowledge_query)
            .await
            .map_err(|e| format!("Knowledge query failed: {}", e))?;
        let known: Vec<KnowledgeRow> = resp
            .take(0)
            .map_err(|e| format!("Failed to parse knowledge: {}", e))?;

        // Get all knowledge in the world (to find blind spots)
        let all_query = "SELECT meta::id(in) AS knower, meta::id(out) AS target, \
                         certainty, out.fact AS fact, learned_at \
                         FROM knows ORDER BY learned_at DESC";
        let mut resp = self
            .db
            .query(all_query)
            .await
            .map_err(|e| format!("World knowledge query failed: {}", e))?;

        #[derive(Deserialize)]
        struct WorldKnowledgeRow {
            knower: String,
            target: String,
            certainty: String,
            fact: Option<String>,
        }

        let all_knowledge: Vec<WorldKnowledgeRow> = resp
            .take(0)
            .map_err(|e| format!("Failed to parse world knowledge: {}", e))?;

        // Classify this character's knowledge
        let mut confident_knowledge = Vec::new();
        let mut suspicions = Vec::new();
        let mut false_beliefs = Vec::new();

        let char_suffix = character_id
            .strip_prefix("character:")
            .unwrap_or(character_id);

        for k in &known {
            let fact_display = k.fact.as_deref().unwrap_or(&k.target);
            match k.certainty.as_str() {
                "Knows" => confident_knowledge.push(fact_display.to_string()),
                "Suspects" | "Uncertain" | "Assumes" => {
                    suspicions.push(fact_display.to_string());
                }
                "BelievesWrongly" => {
                    false_beliefs.push(format!(
                        "{} (truth: {})",
                        fact_display,
                        k.truth_value.as_deref().unwrap_or("unknown")
                    ));
                }
                "Denies" => {
                    false_beliefs.push(format!("{} (denies)", fact_display));
                }
                _ => {}
            }
        }

        // Find blind spots: things others know that this character doesn't
        let known_targets: std::collections::HashSet<&str> =
            known.iter().map(|k| k.target.as_str()).collect();

        let mut blind_spots: Vec<String> = Vec::new();
        let mut seen_targets = std::collections::HashSet::new();

        for wk in &all_knowledge {
            if wk.knower == char_suffix || wk.knower == character_id {
                continue;
            }
            if known_targets.contains(wk.target.as_str()) {
                continue;
            }
            if wk.certainty == "Knows" && seen_targets.insert(wk.target.clone()) {
                let fact_display = wk.fact.as_deref().unwrap_or(&wk.target);
                blind_spots.push(format!("{} (known by {})", fact_display, wk.knower));
                if blind_spots.len() >= 20 {
                    break;
                }
            }
        }

        // Build report
        let mut content_parts = vec![format!("# Knowledge Gap Analysis: {}", character_name)];

        content_parts.push(format!(
            "\n**Summary:** {} known facts, {} suspicions, {} false beliefs, {} blind spots",
            confident_knowledge.len(),
            suspicions.len(),
            false_beliefs.len(),
            blind_spots.len()
        ));

        if !confident_knowledge.is_empty() {
            content_parts.push(format!(
                "\n## Confident Knowledge ({})",
                confident_knowledge.len()
            ));
            for k in &confident_knowledge[..confident_knowledge.len().min(15)] {
                content_parts.push(format!("- {}", k));
            }
            if confident_knowledge.len() > 15 {
                content_parts.push(format!("  ... and {} more", confident_knowledge.len() - 15));
            }
        }

        if !suspicions.is_empty() {
            content_parts.push(format!(
                "\n## Suspicions & Uncertain ({})",
                suspicions.len()
            ));
            for s in &suspicions {
                content_parts.push(format!("- {}", s));
            }
        }

        if !false_beliefs.is_empty() {
            content_parts.push(format!("\n## False Beliefs ({})", false_beliefs.len()));
            for f in &false_beliefs {
                content_parts.push(format!("- {}", f));
            }
        }

        if !blind_spots.is_empty() {
            content_parts.push(format!(
                "\n## Blind Spots — Things Others Know ({})",
                blind_spots.len()
            ));
            for b in &blind_spots {
                content_parts.push(format!("- {}", b));
            }
        }

        let content = content_parts.join("\n");
        let token_estimate = content.len() / 4 + 50;

        Ok(QueryResponse {
            results: vec![EntityResult {
                id: format!("report:knowledge_gap:{}", character_id),
                entity_type: "report".to_string(),
                name: format!("Knowledge Gap: {}", character_name),
                content,
                confidence: None,
                last_modified: None,
            }],
            total: 1,
            next_cursor: None,
            hints: vec![
                "Use dramatic_irony prompt for scene-level tension from these gaps".to_string(),
                "Use knowledge_asymmetries for pairwise comparison".to_string(),
            ],
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_relationship_strength_map(
        &self,
        character_id: &str,
        limit: Option<usize>,
    ) -> Result<QueryResponse, String> {
        let limit = limit.unwrap_or(20).min(MAX_LIMIT);

        // Get character name
        #[derive(Deserialize)]
        struct NameRow {
            name: String,
        }

        let name_query = format!("SELECT name FROM {} LIMIT 1", character_id);
        let mut resp = self
            .db
            .query(&name_query)
            .await
            .map_err(|e| format!("Failed to look up character: {}", e))?;
        let name_row: Option<NameRow> = resp
            .take(0)
            .map_err(|e| format!("Failed to parse character: {}", e))?;
        let character_name = name_row
            .map(|r| r.name)
            .unwrap_or_else(|| character_id.to_string());

        // Get outgoing relationships
        #[derive(Deserialize)]
        struct RelRow {
            other: String,
            other_name: Option<String>,
            rel_type: String,
            subtype: Option<String>,
            label: Option<String>,
        }

        let out_query = format!(
            "SELECT meta::id(out) AS other, out.name AS other_name, \
             type AS rel_type, subtype, label \
             FROM relates_to WHERE in = {} LIMIT {}",
            character_id, limit
        );
        let mut resp = self
            .db
            .query(&out_query)
            .await
            .map_err(|e| format!("Outgoing relationships query failed: {}", e))?;
        let outgoing: Vec<RelRow> = resp
            .take(0)
            .map_err(|e| format!("Failed to parse relationships: {}", e))?;

        // Get incoming relationships
        let in_query = format!(
            "SELECT meta::id(in) AS other, in.name AS other_name, \
             type AS rel_type, subtype, label \
             FROM relates_to WHERE out = {} LIMIT {}",
            character_id, limit
        );
        let mut resp = self
            .db
            .query(&in_query)
            .await
            .map_err(|e| format!("Incoming relationships query failed: {}", e))?;
        let incoming: Vec<RelRow> = resp
            .take(0)
            .map_err(|e| format!("Failed to parse relationships: {}", e))?;

        // Get perceptions (for tension overlay)
        #[derive(Deserialize)]
        struct PercRow {
            other: String,
            tension_level: Option<i32>,
            feelings: Option<String>,
        }

        let perc_query = format!(
            "SELECT meta::id(out) AS other, tension_level, feelings \
             FROM perceives WHERE in = {}",
            character_id
        );
        let mut resp = self
            .db
            .query(&perc_query)
            .await
            .map_err(|e| format!("Perceptions query failed: {}", e))?;
        let perceptions: Vec<PercRow> = resp
            .take(0)
            .map_err(|e| format!("Failed to parse perceptions: {}", e))?;

        // Build perception lookup
        let perc_map: std::collections::HashMap<&str, &PercRow> =
            perceptions.iter().map(|p| (p.other.as_str(), p)).collect();

        // Build output
        let mut content_parts = vec![format!(
            "# Relationship Map: {} ({} outgoing, {} incoming)",
            character_name,
            outgoing.len(),
            incoming.len()
        )];

        if !outgoing.is_empty() {
            content_parts.push("\n## Outgoing Relationships (this character → other)".to_string());
            content_parts
                .push("| Other | Type | Subtype | Label | Tension | Feelings |".to_string());
            content_parts
                .push("|-------|------|---------|-------|---------|----------|".to_string());

            for r in &outgoing {
                let other_display = r.other_name.as_deref().unwrap_or(&r.other);
                let perc = perc_map.get(r.other.as_str());
                content_parts.push(format!(
                    "| {} | {} | {} | {} | {} | {} |",
                    other_display,
                    r.rel_type,
                    r.subtype.as_deref().unwrap_or("-"),
                    r.label.as_deref().unwrap_or("-"),
                    perc.and_then(|p| p.tension_level)
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "-".to_string()),
                    perc.and_then(|p| p.feelings.as_deref()).unwrap_or("-"),
                ));
            }
        }

        if !incoming.is_empty() {
            content_parts.push("\n## Incoming Relationships (other → this character)".to_string());
            content_parts.push("| Other | Type | Subtype | Label |".to_string());
            content_parts.push("|-------|------|---------|-------|".to_string());

            for r in &incoming {
                let other_display = r.other_name.as_deref().unwrap_or(&r.other);
                content_parts.push(format!(
                    "| {} | {} | {} | {} |",
                    other_display,
                    r.rel_type,
                    r.subtype.as_deref().unwrap_or("-"),
                    r.label.as_deref().unwrap_or("-"),
                ));
            }
        }

        if outgoing.is_empty() && incoming.is_empty() {
            content_parts.push("\nNo relationships found for this character.".to_string());
        }

        let content = content_parts.join("\n");
        let token_estimate = content.len() / 4 + 50;

        Ok(QueryResponse {
            results: vec![EntityResult {
                id: format!("report:relationships:{}", character_id),
                entity_type: "report".to_string(),
                name: format!("Relationship Map: {}", character_name),
                content,
                confidence: None,
                last_modified: None,
            }],
            total: 1,
            next_cursor: None,
            hints: vec![
                "Use character_dossier for a full profile including knowledge".to_string(),
                "Use tension_matrix for a world-wide tension overview".to_string(),
            ],
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_narrative_threads(
        &self,
        status: Option<&str>,
        limit: Option<usize>,
    ) -> Result<QueryResponse, String> {
        let limit = limit.unwrap_or(20).min(MAX_LIMIT);
        let filter_status = status.unwrap_or("all");

        let mut threads: Vec<String> = Vec::new();

        // Thread type 1: Knowledge secrets (one character knows, others don't)
        if filter_status == "all" || filter_status == "open" {
            #[derive(Deserialize)]
            struct SecretRow {
                knower: String,
                target: String,
                fact: Option<String>,
            }

            // Find knowledge held by only one character (potential secrets)
            let secret_query = "SELECT meta::id(in) AS knower, meta::id(out) AS target, \
                               out.fact AS fact \
                               FROM knows WHERE certainty = 'Knows' \
                               GROUP BY target \
                               HAVING count() = 1 \
                               LIMIT 20";
            let mut resp = self
                .db
                .query(secret_query)
                .await
                .map_err(|e| format!("Secret knowledge query failed: {}", e))?;
            let secrets: Vec<SecretRow> = resp.take(0).unwrap_or_default();

            for s in &secrets {
                let fact_display = s.fact.as_deref().unwrap_or(&s.target);
                threads.push(format!(
                    "[Secret] Only {} knows: {}",
                    s.knower, fact_display
                ));
                if threads.len() >= limit {
                    break;
                }
            }
        }

        // Thread type 2: False beliefs (unresolved contradictions)
        if (filter_status == "all" || filter_status == "open") && threads.len() < limit {
            #[derive(Deserialize)]
            struct ConflictRow {
                character: String,
                target: String,
                truth_value: Option<String>,
            }

            let conflict_query =
                "SELECT meta::id(in) AS character, meta::id(out) AS target, truth_value \
                 FROM knows WHERE certainty = 'BelievesWrongly' LIMIT 20";
            let mut resp = self
                .db
                .query(conflict_query)
                .await
                .map_err(|e| format!("False beliefs query failed: {}", e))?;
            let conflicts: Vec<ConflictRow> = resp.take(0).unwrap_or_default();

            for c in &conflicts {
                threads.push(format!(
                    "[False Belief] {} believes wrongly about {} (truth: {})",
                    c.character,
                    c.target,
                    c.truth_value.as_deref().unwrap_or("unknown")
                ));
                if threads.len() >= limit {
                    break;
                }
            }
        }

        // Thread type 3: High-tension unresolved relationships
        if (filter_status == "all" || filter_status == "open") && threads.len() < limit {
            #[derive(Deserialize)]
            struct HighTensionRow {
                observer: String,
                target: String,
                tension_level: i32,
                feelings: Option<String>,
            }

            let tension_query = "SELECT meta::id(in) AS observer, meta::id(out) AS target, \
                                tension_level, feelings \
                                FROM perceives WHERE tension_level >= 7 \
                                ORDER BY tension_level DESC LIMIT 10";
            let mut resp = self
                .db
                .query(tension_query)
                .await
                .map_err(|e| format!("High tension query failed: {}", e))?;
            let tensions: Vec<HighTensionRow> = resp.take(0).unwrap_or_default();

            for t in &tensions {
                threads.push(format!(
                    "[High Tension] {} → {} (tension: {}, feels: {})",
                    t.observer,
                    t.target,
                    t.tension_level,
                    t.feelings.as_deref().unwrap_or("-")
                ));
                if threads.len() >= limit {
                    break;
                }
            }
        }

        // Thread type 4: Stale embeddings (entities that changed but haven't been re-embedded)
        if (filter_status == "all" || filter_status == "stale") && threads.len() < limit {
            #[derive(Deserialize)]
            struct StaleRow {
                id: String,
                name: Option<String>,
            }

            let stale_query =
                "SELECT meta::id(id) AS id, name FROM character WHERE embedding_stale = true LIMIT 10";
            let mut resp = self
                .db
                .query(stale_query)
                .await
                .map_err(|e| format!("Stale entities query failed: {}", e))?;
            let stale: Vec<StaleRow> = resp.take(0).unwrap_or_default();

            for s in &stale {
                threads.push(format!(
                    "[Stale] {} needs re-embedding",
                    s.name.as_deref().unwrap_or(&s.id)
                ));
                if threads.len() >= limit {
                    break;
                }
            }
        }

        threads.truncate(limit);

        let mut content_parts = vec![format!(
            "# Narrative Threads ({}, filter: {})",
            threads.len(),
            filter_status
        )];

        if threads.is_empty() {
            content_parts.push("\nNo narrative threads found matching the filter.".to_string());
        } else {
            for t in &threads {
                content_parts.push(format!("- {}", t));
            }
        }

        let content = content_parts.join("\n");
        let token_estimate = content.len() / 4 + 50;

        Ok(QueryResponse {
            results: vec![EntityResult {
                id: "report:narrative_threads".to_string(),
                entity_type: "report".to_string(),
                name: "Narrative Threads".to_string(),
                content,
                confidence: None,
                last_modified: None,
            }],
            total: 1,
            next_cursor: None,
            hints: vec![
                "Use knowledge_gap_analysis for character-specific blind spots".to_string(),
                "Use situation_report for a high-level narrative overview".to_string(),
            ],
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_character_voice(
        &self,
        character_id: &str,
    ) -> Result<QueryResponse, String> {
        // Get character details
        #[derive(Deserialize)]
        struct CharacterRow {
            name: String,
            roles: Option<Vec<String>>,
            description: Option<String>,
            profile: Option<std::collections::HashMap<String, Vec<String>>>,
        }

        let char_query = format!(
            "SELECT name, roles, description, profile FROM {} LIMIT 1",
            character_id
        );
        let mut resp = self
            .db
            .query(&char_query)
            .await
            .map_err(|e| format!("Character lookup failed: {}", e))?;
        let character: Option<CharacterRow> = resp
            .take(0)
            .map_err(|e| format!("Failed to parse character: {}", e))?;

        let character =
            character.ok_or_else(|| format!("Character not found: {}", character_id))?;

        // Get knowledge certainty distribution
        #[derive(Deserialize)]
        struct CertaintyCount {
            certainty: String,
            count: usize,
        }

        let certainty_query = format!(
            "SELECT certainty, count() AS count FROM knows \
             WHERE in = {} GROUP BY certainty",
            character_id
        );
        let mut resp = self
            .db
            .query(&certainty_query)
            .await
            .map_err(|e| format!("Knowledge certainty query failed: {}", e))?;
        let certainty_dist: Vec<CertaintyCount> = resp.take(0).unwrap_or_default();

        // Get perception patterns (how they see others)
        #[derive(Deserialize)]
        struct PercOutRow {
            target_name: Option<String>,
            feelings: Option<String>,
            tension_level: Option<i32>,
        }

        let perc_query = format!(
            "SELECT out.name AS target_name, feelings, tension_level \
             FROM perceives WHERE in = {} LIMIT 10",
            character_id
        );
        let mut resp = self
            .db
            .query(&perc_query)
            .await
            .map_err(|e| format!("Perception query failed: {}", e))?;
        let perceptions: Vec<PercOutRow> = resp.take(0).unwrap_or_default();

        // Build voice profile
        let mut content_parts = vec![format!("# Character Voice: {}", character.name)];

        // Roles & identity
        let roles_display = character
            .roles
            .as_ref()
            .filter(|r| !r.is_empty())
            .map(|r| r.join(", "))
            .unwrap_or_else(|| "unspecified".to_string());
        content_parts.push(format!("**Roles:** {}", roles_display));

        if let Some(desc) = &character.description {
            content_parts.push(format!("**Description:** {}", desc));
        }

        // Profile traits → voice markers
        if let Some(profile) = &character.profile {
            content_parts.push("\n## Psychological Profile → Voice Markers".to_string());
            for (trait_type, values) in profile {
                if values.is_empty() {
                    continue;
                }
                let voice_impact = match trait_type.as_str() {
                    "wounds" => {
                        "Avoids or deflects around these topics. May speak with guarded language."
                    }
                    "desires" => {
                        "Speaks with energy/passion about these. May reveal through enthusiasm."
                    }
                    "contradictions" => {
                        "Speech may be inconsistent on these topics. Tension in word choice."
                    }
                    "fears" => "May use dismissive or overly casual language to mask these.",
                    "secrets" => "Deliberately avoids or redirects conversations near these.",
                    "values" => "Speaks with conviction and certainty about these.",
                    "habits" => {
                        "These manifest as speech patterns, verbal tics, or recurring phrases."
                    }
                    _ => "Informs general speech patterns and word choice.",
                };
                content_parts.push(format!(
                    "- **{}**: {} → *{}*",
                    trait_type,
                    values.join("; "),
                    voice_impact
                ));
            }
        }

        // Knowledge certainty → speech confidence
        content_parts.push("\n## Knowledge State → Speech Confidence".to_string());
        if certainty_dist.is_empty() {
            content_parts.push(
                "- No knowledge recorded. Character speaks from ignorance or assumption."
                    .to_string(),
            );
        } else {
            for c in &certainty_dist {
                let speech_style = match c.certainty.as_str() {
                    "Knows" => "Speaks factually and with authority",
                    "Suspects" => "Uses hedging language: 'I think', 'maybe', 'it seems'",
                    "BelievesWrongly" => {
                        "Speaks confidently but INCORRECTLY — dramatic irony potential"
                    }
                    "Uncertain" => "Qualified statements, questions, seeking confirmation",
                    "Assumes" => "States opinions as fact without evidence",
                    "Denies" => "Actively pushes back, uses negation and dismissal",
                    "Forgotten" => "Vague references, 'I used to know', gaps in conversation",
                    _ => "General speech pattern",
                };
                content_parts.push(format!(
                    "- **{}** ({} facts): {}",
                    c.certainty, c.count, speech_style
                ));
            }
        }

        // Perception patterns → interpersonal voice
        content_parts.push("\n## Perception of Others → Interpersonal Voice".to_string());
        if perceptions.is_empty() {
            content_parts.push(
                "- No perception data. Character's interpersonal style is uncharted.".to_string(),
            );
        } else {
            for p in &perceptions {
                let target = p.target_name.as_deref().unwrap_or("unknown");
                let tension_desc = match p.tension_level {
                    Some(t) if t >= 8 => "hostile, clipped, or explosive",
                    Some(t) if t >= 5 => "guarded, careful word choice",
                    Some(t) if t >= 3 => "somewhat formal or distant",
                    Some(_) => "warm, relaxed, open",
                    None => "neutral",
                };
                content_parts.push(format!(
                    "- **With {}**: {} speech (tension: {}, feels: {})",
                    target,
                    tension_desc,
                    p.tension_level
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "-".to_string()),
                    p.feelings.as_deref().unwrap_or("-"),
                ));
            }
        }

        // Dialogue guidelines
        content_parts.push("\n## Dialogue Guidelines".to_string());
        content_parts.push("### DO".to_string());
        content_parts
            .push("- Reflect knowledge state in what the character references".to_string());
        content_parts
            .push("- Let profile traits color word choice and emotional register".to_string());
        content_parts
            .push("- Adjust formality/tension based on who they're speaking to".to_string());
        content_parts.push("### DON'T".to_string());
        content_parts.push("- Reference facts the character doesn't know".to_string());
        content_parts.push("- Use vocabulary inconsistent with their role/background".to_string());
        content_parts
            .push("- Make them articulate about topics they're in denial about".to_string());

        let content = content_parts.join("\n");
        let token_estimate = content.len() / 4 + 50;

        Ok(QueryResponse {
            results: vec![EntityResult {
                id: format!("report:voice:{}", character_id),
                entity_type: "report".to_string(),
                name: format!("Voice Profile: {}", character.name),
                content,
                confidence: None,
                last_modified: None,
            }],
            total: 1,
            next_cursor: None,
            hints: vec![
                "Use the character_voice prompt for a guided dialogue writing session".to_string(),
                "Use knowledge_gap_analysis to see their blind spots in detail".to_string(),
            ],
            token_estimate,
            truncated: None,
        })
    }
}
