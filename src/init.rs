//! Shared initialization logic for MCP and CLI modes.

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::db::connection::{init_db, load_db_config, DbConfig, NarraDb};
use crate::db::schema::apply_schema;
use crate::embedding::provider::{
    create_embedding_service, load_provider_config, EmbeddingMetadata, ModelMatch,
};
use crate::embedding::{EmbeddingService, StalenessManager};
use crate::repository::{
    SurrealEntityRepository, SurrealKnowledgeRepository, SurrealRelationshipRepository,
};
use crate::services::{
    CachedContextService, CachedSummaryService, ConsistencyChecker, ConsistencyService,
    ContextService, EmotionService, ImpactAnalyzer, ImpactService, SearchService, SummaryService,
    SurrealSearchService,
};
use crate::session::SessionStateManager;

/// Application context holding all services and repositories.
///
/// Shared between MCP server and CLI commands.
pub struct AppContext {
    pub db: Arc<NarraDb>,
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
    pub emotion_service: Arc<dyn EmotionService + Send + Sync>,
    /// Whether the current embedding model mismatches stored world metadata.
    pub embedding_model_mismatch: ModelMatch,
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

        // Load DB config
        let db_config = load_db_config(&data_path);
        match &db_config {
            DbConfig::Embedded { .. } => tracing::info!("Using embedded database"),
            DbConfig::Remote { endpoint, .. } => {
                tracing::info!("Connecting to remote database: {}", endpoint)
            }
        }

        let db = init_db(&db_config, &data_path).await?;
        tracing::info!("Database connected");

        apply_schema(&db).await?;
        tracing::info!("Schema applied");

        let db = Arc::new(db);

        // Session state
        let session_path = data_path.join("session.json");
        let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path)?);
        tracing::info!("Session state loaded");

        // Embedding model — load provider config
        tracing::info!("Initializing embedding model...");
        let provider_config = load_provider_config(&data_path);
        let embedding_service: Arc<dyn EmbeddingService + Send + Sync> =
            create_embedding_service(&provider_config).unwrap_or_else(|e| {
                tracing::error!("Failed to initialize embedding service: {}", e);
                panic!("Embedding service initialization failed");
            });

        if embedding_service.is_available() {
            tracing::info!(
                "Embedding model loaded: {} via {} ({} dimensions)",
                embedding_service.model_id(),
                embedding_service.provider_name(),
                embedding_service.dimensions()
            );
        } else {
            tracing::warn!("Embedding model not available");
        }

        // Check model match against stored world metadata
        let embedding_model_mismatch =
            check_embedding_metadata(&db, embedding_service.as_ref()).await;

        match &embedding_model_mismatch {
            ModelMatch::Mismatch {
                stored_model,
                current_model,
                ..
            } => {
                tracing::warn!(
                    "Embedding model mismatch: world was embedded with '{}', current model is '{}'. \
                     Semantic search may be unreliable. Run 'narra world backfill --force' to re-embed.",
                    stored_model,
                    current_model
                );
            }
            ModelMatch::NoMetadata => {
                tracing::debug!("No embedding metadata stored yet (fresh or pre-metadata world)");
            }
            ModelMatch::Match => {
                tracing::debug!("Embedding model matches stored metadata");
            }
        }

        // Repositories
        let entity_repo = Arc::new(SurrealEntityRepository::new(db.clone()));
        let relationship_repo = Arc::new(SurrealRelationshipRepository::new(db.clone()));
        let knowledge_repo = Arc::new(SurrealKnowledgeRepository::new(db.clone()));

        // Reranker — loads model eagerly, degrades gracefully if unavailable.
        let reranker: Option<Arc<dyn crate::embedding::reranker::RerankerService + Send + Sync>> = {
            let reranker = crate::embedding::reranker::LocalRerankerService::new();
            if crate::embedding::reranker::RerankerService::is_available(&reranker) {
                Some(Arc::new(reranker))
            } else {
                None
            }
        };

        // Services
        let mut search = SurrealSearchService::new(db.clone(), embedding_service.clone());
        if let Some(ref r) = reranker {
            search = search.with_reranker(r.clone());
        }
        let search_service: Arc<dyn SearchService + Send + Sync> = Arc::new(search);
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

        // Emotion classifier — loads model eagerly, degrades gracefully if unavailable.
        let emotion_service: Arc<dyn EmotionService + Send + Sync> = {
            let service = crate::services::LocalEmotionService::new(db.clone());
            if service.is_available() {
                Arc::new(service)
            } else {
                tracing::info!(
                    "Emotion classifier not available, using noop (emotion queries will return errors)"
                );
                Arc::new(crate::services::NoopEmotionService::new())
            }
        };

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
            emotion_service,
            embedding_model_mismatch,
        })
    }
}

/// Check if the current embedding model matches what's stored in world_meta.
async fn check_embedding_metadata(
    db: &NarraDb,
    embedding_service: &dyn EmbeddingService,
) -> ModelMatch {
    let query = "SELECT * FROM world_meta:default";
    let result = db.query(query).await;

    match result {
        Ok(mut response) => {
            let meta: Option<EmbeddingMetadata> = response.take(0).unwrap_or(None);
            match meta {
                Some(stored) => {
                    let current_model = embedding_service.model_id();
                    let current_dimensions = embedding_service.dimensions();

                    if stored.embedding_model == current_model
                        && stored.embedding_dimensions == current_dimensions
                    {
                        ModelMatch::Match
                    } else {
                        ModelMatch::Mismatch {
                            stored_model: stored.embedding_model,
                            stored_dimensions: stored.embedding_dimensions,
                            current_model: current_model.to_string(),
                            current_dimensions,
                        }
                    }
                }
                None => ModelMatch::NoMetadata,
            }
        }
        Err(e) => {
            tracing::debug!("Could not query world_meta: {}", e);
            ModelMatch::NoMetadata
        }
    }
}

/// Update world_meta with the current embedding model info.
pub async fn update_embedding_metadata(
    db: &NarraDb,
    embedding_service: &dyn EmbeddingService,
) -> Result<(), crate::NarraError> {
    db.query(
        "UPSERT world_meta:default SET \
         embedding_model = $model, \
         embedding_dimensions = $dims, \
         embedding_provider = $provider, \
         last_backfill_at = time::now(), \
         updated_at = time::now()",
    )
    .bind(("model", embedding_service.model_id().to_string()))
    .bind(("dims", embedding_service.dimensions() as i64))
    .bind(("provider", embedding_service.provider_name().to_string()))
    .await?;

    Ok(())
}
