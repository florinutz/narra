//! Consistency issues MCP resource.
//!
//! Exposes current consistency violations across all entities as a resource.
//! This is on-demand validation (runs when resource is read).

use crate::db::connection::NarraDb;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::services::consistency::{
    generate_suggested_fix, ConsistencyService, ConsistencySeverity, Violation,
};

/// Summary of consistency issues for resource response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyReport {
    /// Total number of issues found
    pub total_issues: usize,
    /// Number of critical (blocking) issues
    pub critical_count: usize,
    /// Number of warning issues
    pub warning_count: usize,
    /// Number of info issues
    pub info_count: usize,
    /// Issues grouped by severity
    pub issues: ConsistencyIssuesByLevel,
}

/// Issues grouped by severity level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyIssuesByLevel {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub critical: Vec<IssueDetail>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warning: Vec<IssueDetail>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub info: Vec<IssueDetail>,
}

/// Detail for a single consistency issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueDetail {
    /// Entity or relationship this issue relates to
    pub source: String,
    /// Human-readable description
    pub message: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Suggested fix if available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_fix: Option<String>,
}

impl From<&Violation> for IssueDetail {
    fn from(v: &Violation) -> Self {
        Self {
            source: v.fact_id.clone(),
            message: v.message.clone(),
            confidence: v.confidence,
            suggested_fix: generate_suggested_fix(v),
        }
    }
}

/// Get consistency issues as JSON string for MCP resource.
///
/// Queries all characters for timeline and relationship violations.
/// Note: This is on-demand validation, may take a moment for large worlds.
pub async fn get_consistency_issues_resource(
    db: &Arc<NarraDb>,
    consistency_service: &Arc<dyn ConsistencyService>,
) -> Result<String, String> {
    let mut all_violations: Vec<(ConsistencySeverity, Violation)> = Vec::new();

    // Get all characters to check
    let characters: Vec<surrealdb::sql::Thing> = db
        .query("SELECT VALUE id FROM character")
        .await
        .map_err(|e| format!("Failed to query characters: {}", e))?
        .take(0)
        .map_err(|e| format!("Failed to parse characters: {}", e))?;

    // Check each character for violations
    for char_thing in characters {
        let char_id = char_thing.to_string();

        // Timeline violations
        if let Ok(timeline_v) = consistency_service
            .check_timeline_violations(&char_id)
            .await
        {
            for v in timeline_v {
                all_violations.push((v.severity, v));
            }
        }

        // Relationship violations
        if let Ok(rel_v) = consistency_service
            .check_relationship_violations(&char_id)
            .await
        {
            for v in rel_v {
                all_violations.push((v.severity, v));
            }
        }
    }

    // Build report
    let mut critical = Vec::new();
    let mut warning = Vec::new();
    let mut info = Vec::new();

    for (severity, violation) in &all_violations {
        let detail = IssueDetail::from(violation);
        match severity {
            ConsistencySeverity::Critical => critical.push(detail),
            ConsistencySeverity::Warning => warning.push(detail),
            ConsistencySeverity::Info => info.push(detail),
        }
    }

    let report = ConsistencyReport {
        total_issues: all_violations.len(),
        critical_count: critical.len(),
        warning_count: warning.len(),
        info_count: info.len(),
        issues: ConsistencyIssuesByLevel {
            critical,
            warning,
            info,
        },
    };

    serde_json::to_string_pretty(&report)
        .map_err(|e| format!("Failed to serialize consistency report: {}", e))
}
