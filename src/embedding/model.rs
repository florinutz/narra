//! Local embedding model implementation using fastembed.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use tracing::warn;

use crate::embedding::EmbeddingService;
use crate::NarraError;

/// Configuration for embedding model initialization.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Model to use (defaults to BGE-small-en-v1.5)
    pub model: EmbeddingModel,
    /// Optional cache directory for model files
    pub cache_dir: Option<String>,
    /// Show download progress (default: true)
    pub show_download_progress: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: EmbeddingModel::BGESmallENV15,
            cache_dir: None,
            show_download_progress: true,
        }
    }
}

/// Local embedding service using fastembed.
///
/// Wraps TextEmbedding in Arc<Mutex<>> for shared mutable async access.
/// Uses spawn_blocking to offload CPU-intensive embedding operations.
pub struct LocalEmbeddingService {
    model: Option<Arc<Mutex<TextEmbedding>>>,
    available: bool,
    dimensions: usize,
}

impl LocalEmbeddingService {
    /// Create a new local embedding service.
    ///
    /// Attempts to load the specified model. If loading fails (e.g., no internet
    /// on first run), the service will be unavailable but will not error.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for model initialization
    ///
    /// # Returns
    ///
    /// A new LocalEmbeddingService instance (may be unavailable if model load failed).
    pub fn new(config: EmbeddingConfig) -> Result<Self, NarraError> {
        // Determine dimensions before moving config.model
        let dimensions = match config.model {
            EmbeddingModel::BGESmallENV15 => 384,
            _ => 384, // Default to 384 for now
        };

        let mut init_options = InitOptions::new(config.model)
            .with_show_download_progress(config.show_download_progress);

        if let Some(cache_dir) = config.cache_dir {
            init_options = init_options.with_cache_dir(cache_dir.into());
        }

        match TextEmbedding::try_new(init_options) {
            Ok(embedding) => Ok(Self {
                model: Some(Arc::new(Mutex::new(embedding))),
                available: true,
                dimensions,
            }),
            Err(e) => {
                warn!(
                    "Failed to load embedding model: {}. Embedding service will be unavailable.",
                    e
                );
                Ok(Self {
                    model: None,
                    available: false,
                    dimensions, // Use same dimensions even when unavailable
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

        // Clone the Arc to move into spawn_blocking
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| NarraError::Database("Embedding model not loaded".to_string()))?
            .clone();

        let text = text.to_string();

        // Use spawn_blocking since fastembed operations are synchronous and CPU-bound
        let result = tokio::task::spawn_blocking(move || {
            let mut model_guard = model
                .lock()
                .map_err(|e| anyhow::anyhow!("Mutex lock failed: {}", e))?;
            let embeddings = model_guard.embed(vec![text], None)?;
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

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| NarraError::Database("Embedding model not loaded".to_string()))?
            .clone();

        let texts = texts.to_vec();

        // Use spawn_blocking since fastembed operations are synchronous and CPU-bound
        // Use batch size of 50 for optimal performance
        let result = tokio::task::spawn_blocking(move || {
            let mut model_guard = model
                .lock()
                .map_err(|e| anyhow::anyhow!("Mutex lock failed: {}", e))?;
            let embeddings = model_guard.embed(texts, Some(50))?;
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
}
