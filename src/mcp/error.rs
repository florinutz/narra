use crate::NarraError;
use rmcp::model::{Content, ErrorCode, ErrorData, IntoContents};
use serde::Serialize;
use std::borrow::Cow;

/// Structured error response for MCP tool calls.
/// Provides error_code + suggestion so LLMs can auto-fix.
#[derive(Debug, Serialize)]
pub struct ToolError {
    pub error_code: String,
    pub message: String,
    pub suggestion: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub example: Option<serde_json::Value>,
}

impl IntoContents for ToolError {
    fn into_contents(self) -> Vec<Content> {
        let json = serde_json::to_string(&self).unwrap_or_else(|_| self.message.clone());
        vec![Content::text(json)]
    }
}

impl From<String> for ToolError {
    fn from(msg: String) -> Self {
        if msg.contains("not found") {
            ToolError {
                error_code: "NOT_FOUND".into(),
                message: msg,
                suggestion: "Check entity ID format (table:key, e.g. character:alice). Use search to find entities.".into(),
                field: None,
                example: Some(serde_json::json!({ "entity_id": "character:alice" })),
            }
        } else if msg.starts_with("Invalid") {
            ToolError {
                error_code: "INVALID_PARAMS".into(),
                message: msg,
                suggestion: "Check parameter format and valid values.".into(),
                field: None,
                example: None,
            }
        } else if msg.starts_with("CRITICAL") {
            ToolError {
                error_code: "CONSTRAINT_VIOLATION".into(),
                message: msg,
                suggestion: "A universe fact blocks this operation. Review the fact or use update_fact to adjust it.".into(),
                field: None,
                example: None,
            }
        } else if msg.starts_with("Validation") {
            ToolError {
                error_code: "VALIDATION_ERROR".into(),
                message: msg,
                suggestion: "Check field values and required fields.".into(),
                field: None,
                example: None,
            }
        } else if msg.contains("All ") && msg.contains(" failed") {
            ToolError {
                error_code: "BATCH_FAILED".into(),
                message: msg,
                suggestion: "All items in the batch failed. Check individual items for errors."
                    .into(),
                field: None,
                example: None,
            }
        } else {
            ToolError {
                error_code: "INTERNAL_ERROR".into(),
                message: msg,
                suggestion: "Retry the operation or simplify the request.".into(),
                field: None,
                example: None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_found_classification() {
        let err = ToolError::from("Character 'alice' not found".to_string());
        assert_eq!(err.error_code, "NOT_FOUND");
        assert!(err.suggestion.contains("entity ID"));
        assert!(err.example.is_some());
    }

    #[test]
    fn test_invalid_params_classification() {
        let err = ToolError::from("Invalid entity type: foobar".to_string());
        assert_eq!(err.error_code, "INVALID_PARAMS");
    }

    #[test]
    fn test_constraint_violation_classification() {
        let err = ToolError::from("CRITICAL: Universe fact prohibits this action".to_string());
        assert_eq!(err.error_code, "CONSTRAINT_VIOLATION");
        assert!(err.suggestion.contains("universe fact"));
    }

    #[test]
    fn test_validation_error_classification() {
        let err = ToolError::from("Validation failed: name is required".to_string());
        assert_eq!(err.error_code, "VALIDATION_ERROR");
    }

    #[test]
    fn test_batch_failed_classification() {
        let err = ToolError::from("All 3 knowledge entries failed: err1; err2".to_string());
        assert_eq!(err.error_code, "BATCH_FAILED");
    }

    #[test]
    fn test_internal_error_fallback() {
        let err = ToolError::from("Something unexpected happened".to_string());
        assert_eq!(err.error_code, "INTERNAL_ERROR");
        assert_eq!(err.message, "Something unexpected happened");
    }

    #[test]
    fn test_into_contents_produces_json() {
        let err = ToolError::from("Entity not found".to_string());
        let contents = err.into_contents();
        assert_eq!(contents.len(), 1);
    }
}

impl From<NarraError> for ErrorData {
    fn from(err: NarraError) -> Self {
        match err {
            NarraError::NotFound { entity_type, id } => ErrorData {
                code: ErrorCode::INVALID_PARAMS,
                message: Cow::Owned(format!("{} '{}' not found", entity_type, id)),
                data: Some(serde_json::json!({
                    "entity_type": entity_type,
                    "id": id,
                    "suggestion": "Check entity ID or use search to find similar entities"
                })),
            },
            NarraError::Query { message, .. } => ErrorData {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::Owned(format!("Query execution failed: {}", message)),
                data: Some(serde_json::json!({
                    "suggestion": "This may be a temporary issue. Try again or simplify your query."
                })),
            },
            NarraError::Validation(msg) => ErrorData {
                code: ErrorCode::INVALID_PARAMS,
                message: Cow::Owned(format!("Validation failed: {}", msg)),
                data: Some(serde_json::json!({
                    "suggestion": "Check input values match expected formats"
                })),
            },
            NarraError::Database(msg) => ErrorData {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::Owned(format!("Database error: {}", msg)),
                data: Some(serde_json::json!({
                    "suggestion": "This may be a temporary database issue. Try again."
                })),
            },
            NarraError::Conflict(msg) => ErrorData {
                code: ErrorCode::INVALID_PARAMS,
                message: Cow::Owned(format!("Conflict: {}", msg)),
                data: Some(serde_json::json!({
                    "suggestion": "Check for duplicate names or concurrent modifications"
                })),
            },
            NarraError::Transaction(msg) => ErrorData {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::Owned(format!("Transaction error: {}", msg)),
                data: Some(serde_json::json!({
                    "suggestion": "This transaction failed. Try again."
                })),
            },
            NarraError::ReferentialIntegrityViolation {
                entity_type,
                entity_id,
                message,
            } => ErrorData {
                code: ErrorCode::INVALID_PARAMS,
                message: Cow::Owned(format!(
                    "Cannot delete {} '{}': {}",
                    entity_type, entity_id, message
                )),
                data: Some(serde_json::json!({
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "reason": message,
                    "suggestion": "Update or delete the referencing entities before attempting this deletion"
                })),
            },
        }
    }
}
