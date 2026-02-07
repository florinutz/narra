use crate::NarraError;
use rmcp::model::{ErrorCode, ErrorData};
use std::borrow::Cow;

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
