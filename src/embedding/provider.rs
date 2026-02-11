//! Embedding provider configuration and factory.
//!
//! Supports multiple embedding backends via a tagged enum configuration.
//! Default is local fastembed (BGE-small-en-v1.5). API providers available
//! behind the `api-embeddings` feature flag.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::embedding::{EmbeddingConfig, EmbeddingService, LocalEmbeddingService};
use crate::NarraError;

/// Embedding provider configuration.
///
/// Determines which embedding backend to use. Loaded from
/// `{data_path}/embedding.toml` or `NARRA_EMBEDDING_PROVIDER` env var.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum EmbeddingProviderConfig {
    /// Local fastembed model (default).
    Local {
        /// Model name (default: "bge-small-en-v1.5")
        #[serde(default = "default_local_model")]
        model: String,
        /// Cache directory for model files
        #[serde(default)]
        cache_dir: Option<String>,
        /// Show download progress bar (default: true)
        #[serde(default = "default_true")]
        show_download_progress: bool,
    },
    // Future: OpenAi, Cohere, Voyage, etc.
    // Gated behind `api-embeddings` feature flag.
}

fn default_local_model() -> String {
    "bge-small-en-v1.5".to_string()
}

fn default_true() -> bool {
    true
}

impl Default for EmbeddingProviderConfig {
    fn default() -> Self {
        Self::Local {
            model: default_local_model(),
            cache_dir: None,
            show_download_progress: true,
        }
    }
}

/// Stored metadata about the active embedding model in a world database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    pub embedding_model: String,
    pub embedding_dimensions: usize,
    pub embedding_provider: String,
    pub last_backfill_at: Option<String>,
}

/// Result of comparing current provider config against stored world metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelMatch {
    /// No metadata stored yet (fresh database or pre-metadata world).
    NoMetadata,
    /// Current model matches stored metadata.
    Match,
    /// Model mismatch — semantic search unreliable until re-embedding.
    Mismatch {
        stored_model: String,
        stored_dimensions: usize,
        current_model: String,
        current_dimensions: usize,
    },
}

/// Load embedding provider config with priority:
/// 1. `{data_path}/embedding.toml` file
/// 2. `NARRA_EMBEDDING_PROVIDER` env var (JSON)
/// 3. Default (local BGE-small-en-v1.5)
pub fn load_provider_config(data_path: &Path) -> EmbeddingProviderConfig {
    // Try file first
    let config_path = data_path.join("embedding.toml");
    if config_path.exists() {
        match std::fs::read_to_string(&config_path) {
            Ok(contents) => match toml::from_str::<EmbeddingProviderConfig>(&contents) {
                Ok(config) => {
                    info!("Loaded embedding config from {}", config_path.display());
                    return config;
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse {}: {}. Using default.",
                        config_path.display(),
                        e
                    );
                }
            },
            Err(e) => {
                tracing::warn!(
                    "Failed to read {}: {}. Using default.",
                    config_path.display(),
                    e
                );
            }
        }
    }

    // Try env var (JSON format)
    if let Ok(json) = std::env::var("NARRA_EMBEDDING_PROVIDER") {
        match serde_json::from_str::<EmbeddingProviderConfig>(&json) {
            Ok(config) => {
                info!("Loaded embedding config from NARRA_EMBEDDING_PROVIDER env");
                return config;
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to parse NARRA_EMBEDDING_PROVIDER: {}. Using default.",
                    e
                );
            }
        }
    }

    EmbeddingProviderConfig::default()
}

/// Create an embedding service from provider configuration.
pub fn create_embedding_service(
    config: &EmbeddingProviderConfig,
) -> Result<Arc<dyn EmbeddingService + Send + Sync>, NarraError> {
    match config {
        EmbeddingProviderConfig::Local {
            model,
            cache_dir,
            show_download_progress,
        } => {
            let embedding_model = match model.as_str() {
                "bge-small-en-v1.5" => fastembed::EmbeddingModel::BGESmallENV15,
                "bge-base-en-v1.5" => fastembed::EmbeddingModel::BGEBaseENV15,
                "bge-large-en-v1.5" => fastembed::EmbeddingModel::BGELargeENV15,
                "bge-small-en-v1.5-q" => fastembed::EmbeddingModel::BGESmallENV15Q,
                "bge-base-en-v1.5-q" => fastembed::EmbeddingModel::BGEBaseENV15Q,
                "bge-large-en-v1.5-q" => fastembed::EmbeddingModel::BGELargeENV15Q,
                other => {
                    return Err(NarraError::Database(format!(
                        "Unknown local embedding model: '{}'. Supported: bge-small-en-v1.5, bge-base-en-v1.5, bge-large-en-v1.5, bge-small-en-v1.5-q, bge-base-en-v1.5-q, bge-large-en-v1.5-q",
                        other
                    )));
                }
            };

            let embedding_config = EmbeddingConfig {
                model: embedding_model,
                cache_dir: cache_dir.clone(),
                show_download_progress: *show_download_progress,
            };

            let service = LocalEmbeddingService::new(embedding_config)?;
            Ok(Arc::new(service))
        }
    }
}
