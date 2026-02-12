//! Progress reporting abstraction for long-running operations.
//!
//! Decouples services from MCP transport while allowing granular progress
//! updates. MCP tools create a `McpProgressReporter` from their `Meta`/`Peer`;
//! CLI handlers and tests use `NoopProgressReporter`.

use std::sync::Arc;

use async_trait::async_trait;

/// Reports progress for long-running operations.
///
/// Progress values are normalized: `current` goes from 0.0 to `total` (default 1.0).
/// Messages provide human-readable step descriptions.
#[async_trait]
pub trait ProgressReporter: Send + Sync {
    /// Report progress. Implementations should be fire-and-forget (never fail the caller).
    async fn report(&self, current: f64, total: f64, message: Option<String>);

    /// Convenience: report a step out of N total steps.
    async fn step(&self, step: usize, total_steps: usize, message: &str) {
        let current = step as f64 / total_steps as f64;
        self.report(current, 1.0, Some(message.to_string())).await;
    }
}

/// No-op reporter for CLI, tests, and tools without progress support.
pub struct NoopProgressReporter;

#[async_trait]
impl ProgressReporter for NoopProgressReporter {
    async fn report(&self, _current: f64, _total: f64, _message: Option<String>) {}
}

/// Shorthand for creating a no-op reporter.
pub fn noop_progress() -> Arc<dyn ProgressReporter> {
    Arc::new(NoopProgressReporter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Test reporter that counts calls.
    struct CountingReporter {
        count: AtomicUsize,
    }

    #[async_trait]
    impl ProgressReporter for CountingReporter {
        async fn report(&self, _current: f64, _total: f64, _message: Option<String>) {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[tokio::test]
    async fn test_noop_reporter_does_nothing() {
        let reporter = NoopProgressReporter;
        reporter.report(0.5, 1.0, Some("test".into())).await;
        reporter.step(1, 3, "step one").await;
        // No panic, no side effects
    }

    #[tokio::test]
    async fn test_counting_reporter() {
        let reporter = CountingReporter {
            count: AtomicUsize::new(0),
        };
        reporter.report(0.0, 1.0, None).await;
        reporter.report(0.5, 1.0, Some("halfway".into())).await;
        reporter.report(1.0, 1.0, Some("done".into())).await;
        assert_eq!(reporter.count.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn test_step_convenience() {
        let reporter = CountingReporter {
            count: AtomicUsize::new(0),
        };
        reporter.step(1, 5, "step 1").await;
        reporter.step(2, 5, "step 2").await;
        assert_eq!(reporter.count.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_noop_progress_arc() {
        let reporter = noop_progress();
        reporter.report(0.0, 1.0, None).await;
        reporter.step(1, 1, "done").await;
    }
}
