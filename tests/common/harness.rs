//! Test harness for database lifecycle management.
//!
//! Provides isolated database instances per test using tempfile.

use std::sync::Arc;
use tempfile::TempDir;

use narra::db::connection::{init_db, DbConfig, NarraDb};
use narra::db::schema::apply_schema;
use narra::embedding::{EmbeddingService, NoopEmbeddingService};
use narra::mcp::NarraServer;
use narra::session::SessionStateManager;

/// Test harness that manages database lifecycle.
///
/// Each TestHarness creates an isolated database in a temporary directory.
/// The database is automatically cleaned up when the harness is dropped.
pub struct TestHarness {
    /// Database connection wrapped in Arc for service sharing
    pub db: Arc<NarraDb>,
    /// Temporary directory (kept alive while harness exists)
    pub temp_dir: TempDir,
}

impl TestHarness {
    /// Create a new test harness with isolated database.
    ///
    /// Panics if database initialization fails (appropriate for tests).
    pub async fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp directory for test database");

        let db_path = temp_dir.path().join("test.db");
        let config = DbConfig::Embedded {
            path: Some(db_path.to_string_lossy().into_owned()),
        };
        let db = init_db(&config, temp_dir.path())
            .await
            .expect("Failed to initialize test database");

        apply_schema(&db)
            .await
            .expect("Failed to apply schema to test database");

        Self {
            db: Arc::new(db),
            temp_dir,
        }
    }

    /// Get the path to the temporary directory.
    ///
    /// Useful for creating additional files (e.g., session state) in the same
    /// isolated directory.
    pub fn temp_path(&self) -> &std::path::Path {
        self.temp_dir.path()
    }
}

/// Create a NarraServer from a TestHarness (common pattern across handler tests).
pub async fn create_test_server(harness: &TestHarness) -> NarraServer {
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(
        SessionStateManager::load_or_create(&session_path)
            .expect("Failed to create session manager"),
    );
    NarraServer::new(
        harness.db.clone(),
        session_manager,
        test_embedding_service(),
    )
    .await
}

/// Create a no-op embedding service for testing.
///
/// Returns an Arc-wrapped NoopEmbeddingService that reports as unavailable.
/// Used in test contexts where embedding functionality is not needed.
pub fn test_embedding_service() -> Arc<dyn EmbeddingService + Send + Sync> {
    Arc::new(NoopEmbeddingService::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_harness_creates_database() {
        let harness = TestHarness::new().await;
        assert!(Arc::strong_count(&harness.db) == 1);
    }
}
