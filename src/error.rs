use thiserror::Error;

/// Custom error type for Narra operations.
#[derive(Debug, Error)]
pub enum NarraError {
    /// Database operation failed.
    #[error("Database error: {0}")]
    Database(String),

    /// Requested entity was not found.
    #[error("Not found: {entity_type} with id '{id}'")]
    NotFound { entity_type: String, id: String },

    /// Input validation failed.
    #[error("Validation error: {0}")]
    Validation(String),

    /// Conflict detected (e.g., duplicate keys, concurrent modifications).
    #[error("Conflict: {0}")]
    Conflict(String),

    /// Transaction operation failed.
    #[error("Transaction error: {0}")]
    Transaction(String),

    /// Query execution failed.
    #[error("Query error: {message}")]
    Query {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Deletion blocked due to referential integrity constraint.
    #[error("Cannot delete {entity_type} '{entity_id}': {message}")]
    ReferentialIntegrityViolation {
        entity_type: String,
        entity_id: String,
        message: String,
    },
}

impl From<surrealdb::Error> for NarraError {
    fn from(err: surrealdb::Error) -> Self {
        NarraError::Database(err.to_string())
    }
}

impl From<serde_json::Error> for NarraError {
    fn from(err: serde_json::Error) -> Self {
        NarraError::Database(format!("JSON serialization error: {}", err))
    }
}

impl From<std::io::Error> for NarraError {
    fn from(err: std::io::Error) -> Self {
        NarraError::Database(format!("I/O error: {}", err))
    }
}
