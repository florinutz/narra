//! Cross-encoder re-ranking for improved search relevance.
//!
//! Re-ranking sees query and document together (cross-encoder) rather than
//! independently (bi-encoder), dramatically improving relevance ordering.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::warn;

use crate::embedding::candle_backend::{download_model, select_device, CrossEncoderReranker};
use crate::NarraError;

/// Service trait for cross-encoder re-ranking.
#[async_trait]
pub trait RerankerService: Send + Sync {
    /// Re-rank candidate documents against a query.
    ///
    /// Returns (original_index, score) pairs sorted by score descending.
    async fn rerank(
        &self,
        query: &str,
        candidates: &[String],
    ) -> Result<Vec<(usize, f32)>, NarraError>;

    /// Check if the reranker model is loaded and available.
    fn is_available(&self) -> bool;
}

const RERANKER_REPO: &str = "BAAI/bge-reranker-base";

/// Local cross-encoder reranker using candle.
pub struct LocalRerankerService {
    reranker: Option<Arc<CrossEncoderReranker>>,
    available: bool,
}

impl Default for LocalRerankerService {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalRerankerService {
    /// Create a new local reranker service.
    ///
    /// Downloads and loads BGE-reranker-base via candle. If model loading fails,
    /// the service will be unavailable but won't error (graceful degradation).
    pub fn new() -> Self {
        let files = match download_model(RERANKER_REPO, None) {
            Ok(files) => files,
            Err(e) => {
                warn!(
                    "Failed to download reranker model: {}. Re-ranking will be unavailable.",
                    e
                );
                return Self {
                    reranker: None,
                    available: false,
                };
            }
        };

        let device = select_device();

        match CrossEncoderReranker::new(&files, device) {
            Ok(reranker) => {
                tracing::info!("Reranker model loaded (bge-reranker-base via candle)");
                Self {
                    reranker: Some(Arc::new(reranker)),
                    available: true,
                }
            }
            Err(e) => {
                warn!(
                    "Failed to load reranker model: {}. Re-ranking will be unavailable.",
                    e
                );
                Self {
                    reranker: None,
                    available: false,
                }
            }
        }
    }
}

#[async_trait]
impl RerankerService for LocalRerankerService {
    async fn rerank(
        &self,
        query: &str,
        candidates: &[String],
    ) -> Result<Vec<(usize, f32)>, NarraError> {
        if !self.available || candidates.is_empty() {
            return Ok(vec![]);
        }

        let reranker = self
            .reranker
            .as_ref()
            .ok_or_else(|| NarraError::Database("Reranker model not loaded".to_string()))?
            .clone();

        let pairs: Vec<(String, String)> = candidates
            .iter()
            .map(|c| (query.to_string(), c.clone()))
            .collect();

        let result = tokio::task::spawn_blocking(move || {
            let scores = reranker.score_pairs(&pairs)?;
            Ok::<Vec<f32>, anyhow::Error>(scores)
        })
        .await
        .map_err(|e| NarraError::Database(format!("Task join error: {}", e)))?
        .map_err(|e| NarraError::Database(format!("Rerank error: {}", e)))?;

        let mut scored: Vec<(usize, f32)> = result.into_iter().enumerate().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored)
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

/// No-op reranker service for testing.
///
/// Always reports as unavailable and returns empty results.
pub struct NoopRerankerService;

impl Default for NoopRerankerService {
    fn default() -> Self {
        Self::new()
    }
}

impl NoopRerankerService {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl RerankerService for NoopRerankerService {
    async fn rerank(
        &self,
        _query: &str,
        _candidates: &[String],
    ) -> Result<Vec<(usize, f32)>, NarraError> {
        Ok(vec![])
    }

    fn is_available(&self) -> bool {
        false
    }
}
