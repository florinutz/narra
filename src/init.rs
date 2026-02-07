//! Shared initialization logic for MCP and CLI modes.

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use surrealdb::{engine::local::Db, Surreal};

use crate::db::{connection::init_db, schema::apply_schema};
use crate::embedding::{
    EmbeddingConfig, EmbeddingService, LocalEmbeddingService, StalenessManager,
};
use crate::repository::{
    SurrealEntityRepository, SurrealKnowledgeRepository, SurrealRelationshipRepository,
};
use crate::services::{
    CachedContextService, CachedSummaryService, ConsistencyChecker, ConsistencyService,
    ContextService, ImpactAnalyzer, ImpactService, SearchService, SummaryService,
    SurrealSearchService,
};
use crate::session::SessionStateManager;

/// Application context holding all services and repositories.
///
/// Shared between MCP server and CLI commands.
pub struct AppContext {
    pub db: Arc<Surreal<Db>>,
    pub data_path: PathBuf,
    pub session_manager: Arc<SessionStateManager>,
    pub embedding_service: Arc<dyn EmbeddingService + Send + Sync>,
    pub search_service: Arc<dyn SearchService + Send + Sync>,
    pub context_service: Arc<dyn ContextService + Send + Sync>,
    pub summary_service: Arc<dyn SummaryService + Send + Sync>,
    pub impact_service: Arc<dyn ImpactService + Send + Sync>,
    pub consistency_service: Arc<dyn ConsistencyService>,
    pub entity_repo: Arc<SurrealEntityRepository>,
    pub relationship_repo: Arc<SurrealRelationshipRepository>,
    pub knowledge_repo: Arc<SurrealKnowledgeRepository>,
    pub staleness_manager: Arc<StalenessManager>,
}

impl AppContext {
    /// Initialize application context.
    ///
    /// Data path priority: explicit path > NARRA_DATA_PATH env > ./.narra (if exists) > ~/.narra
    pub async fn new(explicit_path: Option<PathBuf>) -> Result<Self> {
        let data_path = explicit_path
            .or_else(|| std::env::var("NARRA_DATA_PATH").ok().map(PathBuf::from))
            .or_else(|| {
                let local_path = Path::new(".narra");
                if local_path.exists() && local_path.is_dir() {
                    Some(local_path.to_path_buf())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .map(|h| h.join(".narra"))
                    .unwrap_or_else(|| PathBuf::from(".narra"))
            });

        tracing::info!("Using data path: {}", data_path.display());

        let db = init_db(&data_path.to_string_lossy()).await?;
        tracing::info!("Database connected");

        apply_schema(&db).await?;
        tracing::info!("Schema applied");

        let db = Arc::new(db);

        // Session state
        let session_path = data_path.join("session.json");
        let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path)?);
        tracing::info!("Session state loaded");

        // Embedding model
        tracing::info!("Initializing embedding model...");
        let embedding_config = EmbeddingConfig::default();
        let embedding_service: Arc<dyn EmbeddingService + Send + Sync> = Arc::new(
            LocalEmbeddingService::new(embedding_config).unwrap_or_else(|e| {
                tracing::error!("Failed to initialize embedding service: {}", e);
                panic!("Embedding service initialization failed");
            }),
        );

        if embedding_service.is_available() {
            tracing::info!(
                "Embedding model loaded ({} dimensions)",
                embedding_service.dimensions()
            );
        } else {
            tracing::warn!("Embedding model not available");
        }

        // Repositories
        let entity_repo = Arc::new(SurrealEntityRepository::new(db.clone()));
        let relationship_repo = Arc::new(SurrealRelationshipRepository::new(db.clone()));
        let knowledge_repo = Arc::new(SurrealKnowledgeRepository::new(db.clone()));

        // Services
        let search_service: Arc<dyn SearchService + Send + Sync> = Arc::new(
            SurrealSearchService::new(db.clone(), embedding_service.clone()),
        );
        let summary_service: Arc<dyn SummaryService + Send + Sync> =
            Arc::new(CachedSummaryService::with_defaults(db.clone()));
        let context_service: Arc<dyn ContextService + Send + Sync> = Arc::new(
            CachedContextService::new(db.clone(), session_manager.clone()),
        );
        let impact_service: Arc<dyn ImpactService + Send + Sync> =
            Arc::new(ImpactAnalyzer::new(db.clone()));
        let consistency_service: Arc<dyn ConsistencyService> =
            Arc::new(ConsistencyChecker::new(db.clone()));
        let staleness_manager =
            Arc::new(StalenessManager::new(db.clone(), embedding_service.clone()));

        Ok(Self {
            db,
            data_path,
            session_manager,
            embedding_service,
            search_service,
            context_service,
            summary_service,
            impact_service,
            consistency_service,
            entity_repo,
            relationship_repo,
            knowledge_repo,
            staleness_manager,
        })
    }
}
