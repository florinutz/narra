//! Cross-encoder re-ranking for improved search relevance.
//!
//! Re-ranking sees query and document together (cross-encoder) rather than
//! independently (bi-encoder), dramatically improving relevance ordering.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use tracing::warn;

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

/// Local cross-encoder reranker using fastembed.
pub struct LocalRerankerService {
    model: Option<Arc<Mutex<TextRerank>>>,
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
    /// Uses BGE-reranker-base by default. If model loading fails, the service
    /// will be unavailable but won't error (graceful degradation).
    pub fn new() -> Self {
        let init_options = RerankInitOptions::new(RerankerModel::BGERerankerBase)
            .with_show_download_progress(true);

        match TextRerank::try_new(init_options) {
            Ok(model) => {
                tracing::info!("Reranker model loaded (bge-reranker-base)");
                Self {
                    model: Some(Arc::new(Mutex::new(model))),
                    available: true,
                }
            }
            Err(e) => {
                warn!(
                    "Failed to load reranker model: {}. Re-ranking will be unavailable.",
                    e
                );
                Self {
                    model: None,
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

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| NarraError::Database("Reranker model not loaded".to_string()))?
            .clone();

        let query = query.to_string();
        let candidates = candidates.to_vec();

        let result = tokio::task::spawn_blocking(move || {
            let mut model_guard = model
                .lock()
                .map_err(|e| anyhow::anyhow!("Mutex lock failed: {}", e))?;
            let results = model_guard.rerank(query, candidates.as_slice(), false, None)?;
            Ok::<Vec<fastembed::RerankResult>, anyhow::Error>(results)
        })
        .await
        .map_err(|e| NarraError::Database(format!("Task join error: {}", e)))?
        .map_err(|e| NarraError::Database(format!("Rerank error: {}", e)))?;

        let mut scored: Vec<(usize, f32)> = result.iter().map(|r| (r.index, r.score)).collect();
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
