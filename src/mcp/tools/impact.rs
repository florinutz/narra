use crate::mcp::NarraServer;
use rmcp::handler::server::wrapper::Parameters;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, JsonSchema)]
pub struct ImpactRequest {
    /// The entity ID to analyze impact for
    pub entity_id: String,
    /// Optional: Proposed change description for more accurate analysis
    #[serde(default)]
    pub proposed_change: Option<String>,
    /// Whether to include full details of affected entities (default: false)
    #[serde(default)]
    pub include_details: Option<bool>,
}

#[derive(Serialize, JsonSchema)]
pub struct ImpactResponse {
    /// Overall severity of the change
    pub severity: String,
    /// Number of entities affected
    pub affected_count: usize,
    /// Affected entities grouped by severity
    pub by_severity: BySeverityGroups,
    /// Warning messages for critical impacts
    pub warnings: Vec<String>,
    /// Suggested actions based on analysis
    pub suggestions: Vec<String>,
    /// Full details of affected entities (if requested)
    #[serde(default)]
    pub affected_details: Option<Vec<AffectedEntityDetail>>,
}

#[derive(Serialize, JsonSchema)]
pub struct BySeverityGroups {
    pub critical: Vec<String>,
    pub high: Vec<String>,
    pub medium: Vec<String>,
    pub low: Vec<String>,
}

#[derive(Serialize, JsonSchema)]
pub struct AffectedEntityDetail {
    pub id: String,
    pub entity_type: String,
    pub name: String,
    pub distance: usize,
    pub reason: String,
}

impl NarraServer {
    /// Handler for analyze_impact tool - implementation called from server.rs
    pub async fn handle_analyze_impact(
        &self,
        Parameters(request): Parameters<ImpactRequest>,
    ) -> Result<ImpactResponse, String> {
        let change_desc = request.proposed_change.as_deref().unwrap_or("analyze");
        let analysis = self
            .impact_service
            .analyze_impact(&request.entity_id, change_desc, 3)
            .await
            .map_err(|e| format!("Impact analysis failed: {}", e))?;

        // Group affected entities by severity
        let by_severity = BySeverityGroups {
            critical: analysis
                .affected_by_severity
                .get("critical")
                .map(|v| v.iter().map(|e| e.id.clone()).collect())
                .unwrap_or_default(),
            high: analysis
                .affected_by_severity
                .get("high")
                .map(|v| v.iter().map(|e| e.id.clone()).collect())
                .unwrap_or_default(),
            medium: analysis
                .affected_by_severity
                .get("medium")
                .map(|v| v.iter().map(|e| e.id.clone()).collect())
                .unwrap_or_default(),
            low: analysis
                .affected_by_severity
                .get("low")
                .map(|v| v.iter().map(|e| e.id.clone()).collect())
                .unwrap_or_default(),
        };

        // Generate suggestions based on analysis
        let mut suggestions = vec![];

        if !by_severity.critical.is_empty() {
            suggestions.push(format!(
                "CRITICAL: {} protected or directly connected entities would be affected. Review before proceeding.",
                by_severity.critical.len()
            ));
        }

        if analysis.total_affected > 10 {
            suggestions.push(
                "Consider breaking this change into smaller, incremental mutations.".to_string(),
            );
        }

        if !analysis.warnings.is_empty() {
            suggestions.push("Review warnings below before proceeding.".to_string());
        }

        if !analysis.has_protected_impact && analysis.total_affected < 3 {
            suggestions.push("Low impact change. Safe to proceed.".to_string());
        }

        // Determine overall severity
        let severity = if analysis.has_protected_impact {
            "Critical".to_string()
        } else if analysis.total_affected > 10 {
            "High".to_string()
        } else if analysis.total_affected > 3 {
            "Medium".to_string()
        } else {
            "Low".to_string()
        };

        // Include details if requested
        let affected_details = if request.include_details.unwrap_or(false) {
            let all_affected: Vec<AffectedEntityDetail> = analysis
                .affected_by_severity
                .values()
                .flatten()
                .map(|e| AffectedEntityDetail {
                    id: e.id.clone(),
                    entity_type: e.entity_type.clone(),
                    name: e.name.clone(),
                    distance: e.distance,
                    reason: e.reason.clone(),
                })
                .collect();
            Some(all_affected)
        } else {
            None
        };

        Ok(ImpactResponse {
            severity,
            affected_count: analysis.total_affected,
            by_severity,
            warnings: analysis.warnings,
            suggestions,
            affected_details,
        })
    }

    /// Handler for protect_entity tool - implementation called from server.rs
    pub async fn handle_protect_entity(
        &self,
        Parameters(entity_id): Parameters<String>,
    ) -> Result<String, String> {
        self.impact_service.protect_entity(&entity_id).await;
        Ok(format!("Entity '{}' is now protected", entity_id))
    }

    /// Handler for unprotect_entity tool - implementation called from server.rs
    pub async fn handle_unprotect_entity(
        &self,
        Parameters(entity_id): Parameters<String>,
    ) -> Result<String, String> {
        self.impact_service.unprotect_entity(&entity_id).await;
        Ok(format!("Protection removed from entity '{}'", entity_id))
    }
}
