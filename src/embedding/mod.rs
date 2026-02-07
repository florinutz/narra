//! Embedding infrastructure for semantic search.
//!
//! This module provides text embedding capabilities using local models via fastembed.
//! The EmbeddingService trait abstracts embedding operations for swappability,
//! while LocalEmbeddingService implements it using BGE-small-en-v1.5.

pub mod backfill;
pub mod composite;
pub mod model;
pub mod queries;
pub mod staleness;

use async_trait::async_trait;

use crate::NarraError;

pub use backfill::{BackfillService, BackfillStats};
pub use model::{EmbeddingConfig, LocalEmbeddingService};
pub use staleness::StalenessManager;

/// No-op embedding service for testing.
///
/// Always reports as unavailable and returns errors for embed operations.
/// Used in test contexts where embedding functionality is not needed.
pub struct NoopEmbeddingService;

impl Default for NoopEmbeddingService {
    fn default() -> Self {
        Self::new()
    }
}

impl NoopEmbeddingService {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl EmbeddingService for NoopEmbeddingService {
    async fn embed_text(&self, _text: &str) -> Result<Vec<f32>, NarraError> {
        Err(NarraError::Database(
            "Embedding service is not available (noop)".to_string(),
        ))
    }

    async fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, NarraError> {
        Err(NarraError::Database(
            "Embedding service is not available (noop)".to_string(),
        ))
    }

    fn dimensions(&self) -> usize {
        384 // Match BGE-small-en-v1.5 dimensions
    }

    fn is_available(&self) -> bool {
        false
    }
}

/// Service trait for generating text embeddings.
///
/// Abstracts embedding operations to allow different implementations
/// (local models, external APIs, mock services for testing).
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    /// Generate embedding for a single text string.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    ///
    /// # Returns
    ///
    /// A vector of f32 values representing the embedding.
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, NarraError>;

    /// Generate embeddings for multiple texts in batch.
    ///
    /// More efficient than calling embed_text repeatedly for large batches.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of strings to embed
    ///
    /// # Returns
    ///
    /// A vector of embeddings, one per input text.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, NarraError>;

    /// Get embedding dimensions (e.g., 384 for BGE-small).
    fn dimensions(&self) -> usize;

    /// Check if the embedding model is available.
    ///
    /// Returns false if model failed to load (e.g., no internet on first run).
    fn is_available(&self) -> bool;
}
