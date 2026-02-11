//! Local embedding model implementation using candle.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::warn;

use crate::embedding::candle_backend::{download_model, select_device, BertEmbedder};
use crate::embedding::EmbeddingService;
use crate::NarraError;

/// Configuration for embedding model initialization.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// HuggingFace repo ID (e.g. "BAAI/bge-small-en-v1.5")
    pub model_repo: String,
    /// Embedding dimensions (e.g. 384 for BGE-small)
    pub dimensions: usize,
    /// Short model identifier (e.g. "bge-small-en-v1.5")
    pub model_id: String,
    /// Optional cache directory for model files
    pub cache_dir: Option<String>,
    /// Show download progress (default: true)
    pub show_download_progress: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_repo: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            model_id: "bge-small-en-v1.5".to_string(),
            cache_dir: None,
            show_download_progress: true,
        }
    }
}

/// Local embedding service using candle.
///
/// Wraps BertEmbedder in Arc for shared async access.
/// Uses spawn_blocking to offload CPU/GPU-intensive embedding operations.
pub struct LocalEmbeddingService {
    embedder: Option<Arc<BertEmbedder>>,
    available: bool,
    dimensions: usize,
    model_id: String,
}

impl LocalEmbeddingService {
    /// Create a new local embedding service.
    ///
    /// Downloads model files from HuggingFace Hub (cached after first run),
    /// then loads the BERT model via candle. If loading fails (e.g., no internet
    /// on first run), the service will be unavailable but will not error.
    pub fn new(config: EmbeddingConfig) -> Result<Self, NarraError> {
        let dimensions = config.dimensions;
        let model_id = config.model_id.clone();

        let files = match download_model(
            &config.model_repo,
            config.cache_dir.as_deref().map(std::path::Path::new),
        ) {
            Ok(files) => files,
            Err(e) => {
                warn!(
                    "Failed to download embedding model: {}. Embedding service will be unavailable.",
                    e
                );
                return Ok(Self {
                    embedder: None,
                    available: false,
                    dimensions,
                    model_id,
                });
            }
        };

        let device = select_device();

        match BertEmbedder::new(&files, device) {
            Ok(embedder) => Ok(Self {
                embedder: Some(Arc::new(embedder)),
                available: true,
                dimensions,
                model_id,
            }),
            Err(e) => {
                warn!(
                    "Failed to load embedding model: {}. Embedding service will be unavailable.",
                    e
                );
                Ok(Self {
                    embedder: None,
                    available: false,
                    dimensions,
                    model_id,
                })
            }
        }
    }
}

#[async_trait]
impl EmbeddingService for LocalEmbeddingService {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, NarraError> {
        if !self.available {
            return Err(NarraError::Database(
                "Embedding service is not available".to_string(),
            ));
        }

        let embedder = self
            .embedder
            .as_ref()
            .ok_or_else(|| NarraError::Database("Embedding model not loaded".to_string()))?
            .clone();

        let text = text.to_string();

        // Use spawn_blocking since candle operations are synchronous and CPU/GPU-bound
        let result = tokio::task::spawn_blocking(move || {
            let embeddings = embedder.embed(&[text])?;
            Ok::<Vec<Vec<f32>>, anyhow::Error>(embeddings)
        })
        .await
        .map_err(|e| NarraError::Database(format!("Task join error: {}", e)))?
        .map_err(|e| NarraError::Database(format!("Embedding error: {}", e)))?;

        result
            .into_iter()
            .next()
            .ok_or_else(|| NarraError::Database("No embedding returned".to_string()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, NarraError> {
        if !self.available {
            return Err(NarraError::Database(
                "Embedding service is not available".to_string(),
            ));
        }

        let embedder = self
            .embedder
            .as_ref()
            .ok_or_else(|| NarraError::Database("Embedding model not loaded".to_string()))?
            .clone();

        let texts = texts.to_vec();

        // Use spawn_blocking since candle operations are synchronous and CPU/GPU-bound
        let result = tokio::task::spawn_blocking(move || {
            let embeddings = embedder.embed(&texts)?;
            Ok::<Vec<Vec<f32>>, anyhow::Error>(embeddings)
        })
        .await
        .map_err(|e| NarraError::Database(format!("Task join error: {}", e)))?
        .map_err(|e| NarraError::Database(format!("Embedding error: {}", e)))?;

        Ok(result)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider_name(&self) -> &str {
        "candle"
    }
}
