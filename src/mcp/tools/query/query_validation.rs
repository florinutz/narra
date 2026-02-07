use crate::mcp::NarraServer;
use crate::mcp::{EntityResult, QueryResponse, ValidationIssue};

impl NarraServer {
    // === Consolidated from validate tool ===

    pub(crate) async fn handle_validate_entity_query(
        &self,
        entity_id: &str,
    ) -> Result<QueryResponse, String> {
        use crate::services::generate_suggested_fix;
        let entity_type = entity_id.split(':').next().unwrap_or("unknown");
        let mut all_issues: Vec<ValidationIssue> = Vec::new();

        // Fact violations
        let fact_result = self
            .consistency_service
            .check_entity_mutation(entity_id, &serde_json::json!({}))
            .await
            .map_err(|e| format!("Fact check failed: {}", e))?;

        for violations in fact_result.violations_by_severity.values() {
            for v in violations {
                all_issues.push(ValidationIssue {
                    issue_type: "fact_violation".into(),
                    severity: format!("{:?}", v.severity).to_uppercase(),
                    message: v.message.clone(),
                    suggested_fix: generate_suggested_fix(v),
                    confidence: v.confidence,
                });
            }
        }

        // Timeline and relationship violations (characters only)
        if entity_type == "character" {
            let char_id = entity_id.split(':').nth(1).unwrap_or(entity_id);

            if let Ok(timeline_violations) = self
                .consistency_service
                .check_timeline_violations(char_id)
                .await
            {
                for v in timeline_violations {
                    all_issues.push(ValidationIssue {
                        issue_type: "timeline_violation".into(),
                        severity: format!("{:?}", v.severity).to_uppercase(),
                        message: v.message.clone(),
                        suggested_fix: generate_suggested_fix(&v),
                        confidence: v.confidence,
                    });
                }
            }

            if let Ok(rel_violations) = self
                .consistency_service
                .check_relationship_violations(char_id)
                .await
            {
                for v in rel_violations {
                    all_issues.push(ValidationIssue {
                        issue_type: "relationship_violation".into(),
                        severity: format!("{:?}", v.severity).to_uppercase(),
                        message: v.message.clone(),
                        suggested_fix: generate_suggested_fix(&v),
                        confidence: v.confidence,
                    });
                }
            }
        }

        let summary = if all_issues.is_empty() {
            format!("Validated {} - no issues found", entity_id)
        } else {
            format!(
                "Validated {} - {} issue(s) found",
                entity_id,
                all_issues.len()
            )
        };

        let content = if all_issues.is_empty() {
            "No consistency issues found.".to_string()
        } else {
            all_issues
                .iter()
                .map(|i| {
                    let fix = i.suggested_fix.as_deref().unwrap_or("no fix suggested");
                    format!(
                        "[{}] {} - {} (fix: {})",
                        i.severity, i.issue_type, i.message, fix
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        Ok(QueryResponse {
            results: vec![EntityResult {
                id: entity_id.to_string(),
                entity_type: "validation_report".to_string(),
                name: summary.clone(),
                content,
                confidence: None,
                last_modified: None,
            }],
            total: 1,
            next_cursor: None,
            hints: if all_issues.is_empty() {
                vec!["No consistency issues found.".to_string()]
            } else {
                vec![summary]
            },
            token_estimate: 500,
        })
    }

    pub(crate) async fn handle_investigate_contradictions_query(
        &self,
        entity_id: &str,
        max_depth: usize,
    ) -> Result<QueryResponse, String> {
        use crate::services::generate_suggested_fix;

        let (violations, entities_checked) = self
            .consistency_service
            .investigate_contradictions(entity_id, max_depth)
            .await
            .map_err(|e| format!("Investigation failed: {}", e))?;

        let issues: Vec<ValidationIssue> = violations
            .iter()
            .map(|v| ValidationIssue {
                issue_type: if v.message.contains("timeline")
                    || v.message.contains("before learning")
                {
                    "timeline_violation".into()
                } else if v.message.contains("relationship")
                    || v.message.contains("Circular")
                    || v.message.contains("Asymmetric")
                {
                    "relationship_violation".into()
                } else {
                    "fact_violation".into()
                },
                severity: format!("{:?}", v.severity).to_uppercase(),
                message: v.message.clone(),
                suggested_fix: generate_suggested_fix(v),
                confidence: v.confidence,
            })
            .collect();

        let summary = format!(
            "Investigated {} (depth {}) - {} issue(s) across {} entities",
            entity_id,
            max_depth,
            issues.len(),
            entities_checked
        );

        let content = if issues.is_empty() {
            "No contradictions found.".to_string()
        } else {
            issues
                .iter()
                .map(|i| format!("[{}] {} - {}", i.severity, i.issue_type, i.message))
                .collect::<Vec<_>>()
                .join("\n")
        };

        Ok(QueryResponse {
            results: vec![EntityResult {
                id: entity_id.to_string(),
                entity_type: "investigation_report".to_string(),
                name: summary.clone(),
                content,
                confidence: None,
                last_modified: None,
            }],
            total: entities_checked,
            next_cursor: None,
            hints: vec![summary],
            token_estimate: 500,
        })
    }

    // === Consolidated from analyze_impact tool ===

    pub(crate) async fn handle_analyze_impact_query(
        &self,
        entity_id: &str,
        proposed_change: Option<String>,
        include_details: Option<bool>,
    ) -> Result<QueryResponse, String> {
        let change_desc = proposed_change.as_deref().unwrap_or("analyze");
        let analysis = self
            .impact_service
            .analyze_impact(entity_id, change_desc, 3)
            .await
            .map_err(|e| format!("Impact analysis failed: {}", e))?;

        let severity = if analysis.has_protected_impact {
            "Critical"
        } else if analysis.total_affected > 10 {
            "High"
        } else if analysis.total_affected > 3 {
            "Medium"
        } else {
            "Low"
        };

        let mut content_parts = vec![
            format!("Severity: {}", severity),
            format!("Affected entities: {}", analysis.total_affected),
        ];

        // Group by severity
        for (sev, entities) in &analysis.affected_by_severity {
            if !entities.is_empty() {
                let ids: Vec<&str> = entities.iter().map(|e| e.id.as_str()).collect();
                content_parts.push(format!("{}: {}", sev, ids.join(", ")));
            }
        }

        if include_details.unwrap_or(false) {
            for entities in analysis.affected_by_severity.values() {
                for e in entities {
                    content_parts.push(format!(
                        "  {} ({}) - distance: {}, reason: {}",
                        e.id, e.entity_type, e.distance, e.reason
                    ));
                }
            }
        }

        let mut hints = analysis.warnings.clone();
        if analysis.has_protected_impact {
            hints.push(
                "CRITICAL: Protected entities would be affected. Review before proceeding."
                    .to_string(),
            );
        }
        if analysis.total_affected > 10 {
            hints.push(
                "Consider breaking this change into smaller, incremental mutations.".to_string(),
            );
        }
        if !analysis.has_protected_impact && analysis.total_affected < 3 {
            hints.push("Low impact change. Safe to proceed.".to_string());
        }

        Ok(QueryResponse {
            results: vec![EntityResult {
                id: entity_id.to_string(),
                entity_type: "impact_analysis".to_string(),
                name: format!("Impact Analysis: {} ({})", entity_id, severity),
                content: content_parts.join("\n"),
                confidence: None,
                last_modified: None,
            }],
            total: analysis.total_affected,
            next_cursor: None,
            hints,
            token_estimate: 500,
        })
    }
}
