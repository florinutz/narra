use crate::db::connection::NarraDb;
use async_trait::async_trait;
use moka::future::Cache;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use crate::repository::{EntityRepository, SurrealEntityRepository};
use crate::NarraError;

/// Detail level for entity retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DetailLevel {
    /// Full entity details (no summarization)
    Full,
    /// Summarized for token efficiency (default)
    #[default]
    Summary,
    /// Minimal - just name and type
    Minimal,
}

/// Token threshold configuration for summarization.
#[derive(Debug, Clone)]
pub struct SummaryConfig {
    /// Token threshold above which entities are summarized (default: 200)
    pub summary_threshold: usize,
    /// Target token count for summaries (default: 50)
    pub summary_target: usize,
    /// Cache TTL in seconds (default: 300 = 5 minutes)
    pub cache_ttl_secs: u64,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            summary_threshold: 200,
            summary_target: 50,
            cache_ttl_secs: 300,
        }
    }
}

/// A cached summary for an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySummary {
    /// Entity ID
    pub id: String,
    /// Entity type
    pub entity_type: String,
    /// Display name
    pub name: String,
    /// Summarized content (or full content if under threshold)
    pub content: String,
    /// Whether this is a summary or full content
    pub is_summarized: bool,
    /// Estimated token count
    pub estimated_tokens: usize,
    /// Version/hash of source entity (for invalidation)
    pub source_version: String,
}

/// Full entity content with all fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityFullContent {
    /// Entity ID
    pub id: String,
    /// Entity type
    pub entity_type: String,
    /// Display name
    pub name: String,
    /// Full content with all fields
    pub content: String,
    /// Estimated token count
    pub estimated_tokens: usize,
}

/// Summary service trait for entity summarization.
#[async_trait]
pub trait SummaryService: Send + Sync {
    /// Get entity content at specified detail level.
    ///
    /// - Full: Returns complete entity content
    /// - Summary: Returns summarized content if over threshold, full otherwise
    /// - Minimal: Returns just name and type
    async fn get_entity_content(
        &self,
        entity_id: &str,
        detail_level: DetailLevel,
    ) -> Result<Option<EntitySummary>, NarraError>;

    /// Get full entity content regardless of size.
    /// Use when user explicitly requests full detail.
    async fn get_full_content(
        &self,
        entity_id: &str,
    ) -> Result<Option<EntityFullContent>, NarraError>;

    /// Invalidate cached summary for an entity.
    /// Call this when entity is updated.
    async fn invalidate(&self, entity_id: &str);

    /// Invalidate all cached summaries.
    async fn invalidate_all(&self);

    /// Check if entity exceeds summary threshold.
    async fn needs_summarization(&self, entity_id: &str) -> Result<bool, NarraError>;

    /// Get estimated token count for an entity.
    async fn estimate_tokens(&self, entity_id: &str) -> Result<usize, NarraError>;
}

/// Cached implementation of SummaryService.
pub struct CachedSummaryService {
    entity_repo: Arc<SurrealEntityRepository>,
    /// Cache for entity summaries (id -> EntitySummary)
    summary_cache: Cache<String, EntitySummary>,
    config: SummaryConfig,
}

impl CachedSummaryService {
    /// Create a new cached summary service.
    pub fn new(db: Arc<NarraDb>, config: SummaryConfig) -> Self {
        let summary_cache = Cache::builder()
            .max_capacity(10_000)
            .time_to_live(Duration::from_secs(config.cache_ttl_secs))
            .build();

        Self {
            entity_repo: Arc::new(SurrealEntityRepository::new(db)),
            summary_cache,
            config,
        }
    }

    /// Create with default config.
    pub fn with_defaults(db: Arc<NarraDb>) -> Self {
        Self::new(db, SummaryConfig::default())
    }

    /// Compute a version hash from entity content for cache invalidation.
    fn compute_version(content: &str) -> String {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Estimate tokens for text (simple heuristic: ~4 chars per token).
    fn estimate_tokens_for_text(text: &str) -> usize {
        text.len().div_ceil(4)
    }

    /// Generate a summary for entity content.
    /// Uses simple truncation with ellipsis - could be enhanced with LLM summarization.
    fn generate_summary(&self, full_content: &str) -> String {
        let target_chars = self.config.summary_target * 4; // ~4 chars per token

        if full_content.len() <= target_chars {
            return full_content.to_string();
        }

        // Find a good break point (sentence or word boundary)
        let truncated = &full_content[..target_chars];

        // Try to end at sentence boundary
        if let Some(pos) = truncated.rfind(". ") {
            return format!("{}...", &truncated[..=pos]);
        }

        // Fall back to word boundary
        if let Some(pos) = truncated.rfind(' ') {
            return format!("{}...", &truncated[..pos]);
        }

        format!("{}...", truncated)
    }

    /// Build full content string and name for a character.
    async fn build_character_content(
        &self,
        id: &str,
    ) -> Result<Option<(String, String)>, NarraError> {
        let char = self.entity_repo.get_character(id).await?;

        Ok(char.map(|c| {
            let name = c.name.clone();
            let mut parts = vec![format!("Name: {}", c.name)];

            if !c.aliases.is_empty() {
                parts.push(format!("Aliases: {}", c.aliases.join(", ")));
            }
            if !c.roles.is_empty() {
                parts.push(format!("Roles: {}", c.roles.join(", ")));
            }

            // Profile sections (sorted for deterministic output)
            let mut profile_keys: Vec<&String> = c.profile.keys().collect();
            profile_keys.sort();
            for key in profile_keys {
                if let Some(entries) = c.profile.get(key) {
                    if !entries.is_empty() {
                        let label = key.replace('_', " ");
                        parts.push(format!("{}: {}", label, entries.join("; ")));
                    }
                }
            }

            (parts.join("\n"), name)
        }))
    }

    /// Build full content string and name for a location.
    async fn build_location_content(
        &self,
        id: &str,
    ) -> Result<Option<(String, String)>, NarraError> {
        let loc = self.entity_repo.get_location(id).await?;

        Ok(loc.map(|l| {
            let name = l.name.clone();
            let mut parts = vec![format!("Name: {}", l.name)];

            if let Some(desc) = &l.description {
                parts.push(format!("Description: {}", desc));
            }

            (parts.join("\n"), name)
        }))
    }

    /// Build full content string and name for an event.
    async fn build_event_content(&self, id: &str) -> Result<Option<(String, String)>, NarraError> {
        let event = self.entity_repo.get_event(id).await?;

        Ok(event.map(|e| {
            let name = e.title.clone();
            let mut parts = vec![format!("Title: {}", e.title)];

            if let Some(desc) = &e.description {
                parts.push(format!("Description: {}", desc));
            }
            parts.push(format!("Sequence: {}", e.sequence));
            if let Some(date) = &e.date {
                parts.push(format!("Date: {}", date));
                if let Some(precision) = &e.date_precision {
                    parts.push(format!("Precision: {}", precision));
                }
            }

            (parts.join("\n"), name)
        }))
    }

    /// Build full content string and name for a scene.
    async fn build_scene_content(&self, id: &str) -> Result<Option<(String, String)>, NarraError> {
        let scene = self.entity_repo.get_scene(id).await?;

        Ok(scene.map(|s| {
            let name = s.title.clone();
            let mut parts = vec![format!("Title: {}", s.title)];

            if let Some(summary) = &s.summary {
                parts.push(format!("Summary: {}", summary));
            }
            parts.push(format!("Event: {}", s.event));
            parts.push(format!("Location: {}", s.primary_location));

            (parts.join("\n"), name)
        }))
    }

    /// Get full content for any entity type (single DB fetch per entity).
    async fn get_entity_full_content(
        &self,
        entity_id: &str,
    ) -> Result<Option<(String, String, String)>, NarraError> {
        let parts: Vec<&str> = entity_id.split(':').collect();
        if parts.len() != 2 {
            return Ok(None);
        }

        let (entity_type, id) = (parts[0], parts[1]);

        let content_and_name = match entity_type {
            "character" => self.build_character_content(id).await?,
            "location" => self.build_location_content(id).await?,
            "event" => self.build_event_content(id).await?,
            "scene" => self.build_scene_content(id).await?,
            _ => return Ok(None),
        };

        Ok(content_and_name.map(|(content, name)| (entity_type.to_string(), name, content)))
    }
}

#[async_trait]
impl SummaryService for CachedSummaryService {
    async fn get_entity_content(
        &self,
        entity_id: &str,
        detail_level: DetailLevel,
    ) -> Result<Option<EntitySummary>, NarraError> {
        // Minimal just returns type and name
        if detail_level == DetailLevel::Minimal {
            let (entity_type, name, _) = match self.get_entity_full_content(entity_id).await? {
                Some(data) => data,
                None => return Ok(None),
            };

            return Ok(Some(EntitySummary {
                id: entity_id.to_string(),
                entity_type,
                name: name.clone(),
                content: name,
                is_summarized: true,
                estimated_tokens: 5,
                source_version: String::new(),
            }));
        }

        // Full bypasses cache
        if detail_level == DetailLevel::Full {
            let full = self.get_full_content(entity_id).await?;
            return Ok(full.map(|f| EntitySummary {
                id: f.id,
                entity_type: f.entity_type,
                name: f.name,
                content: f.content,
                is_summarized: false,
                estimated_tokens: f.estimated_tokens,
                source_version: String::new(),
            }));
        }

        // Summary mode - check cache first.
        // Trust TTL + explicit invalidation (callers call invalidate() on mutation).
        // No DB round-trip on cache hit.
        let cache_key = format!("{}:summary", entity_id);
        if let Some(cached) = self.summary_cache.get(&cache_key).await {
            return Ok(Some(cached));
        }

        // Generate summary
        let (entity_type, name, full_content) =
            match self.get_entity_full_content(entity_id).await? {
                Some(data) => data,
                None => return Ok(None),
            };

        let full_tokens = Self::estimate_tokens_for_text(&full_content);
        let version = Self::compute_version(&full_content);

        let summary = if full_tokens > self.config.summary_threshold {
            // Needs summarization
            let summarized_content = self.generate_summary(&full_content);
            let summary_tokens = Self::estimate_tokens_for_text(&summarized_content);

            EntitySummary {
                id: entity_id.to_string(),
                entity_type,
                name,
                content: summarized_content,
                is_summarized: true,
                estimated_tokens: summary_tokens,
                source_version: version,
            }
        } else {
            // Under threshold - return full content
            EntitySummary {
                id: entity_id.to_string(),
                entity_type,
                name,
                content: full_content,
                is_summarized: false,
                estimated_tokens: full_tokens,
                source_version: version,
            }
        };

        // Cache the result
        self.summary_cache.insert(cache_key, summary.clone()).await;

        Ok(Some(summary))
    }

    async fn get_full_content(
        &self,
        entity_id: &str,
    ) -> Result<Option<EntityFullContent>, NarraError> {
        let (entity_type, name, content) = match self.get_entity_full_content(entity_id).await? {
            Some(data) => data,
            None => return Ok(None),
        };

        let estimated_tokens = Self::estimate_tokens_for_text(&content);

        Ok(Some(EntityFullContent {
            id: entity_id.to_string(),
            entity_type,
            name,
            content,
            estimated_tokens,
        }))
    }

    async fn invalidate(&self, entity_id: &str) {
        let cache_key = format!("{}:summary", entity_id);
        self.summary_cache.invalidate(&cache_key).await;
    }

    async fn invalidate_all(&self) {
        self.summary_cache.invalidate_all();
    }

    async fn needs_summarization(&self, entity_id: &str) -> Result<bool, NarraError> {
        let tokens = self.estimate_tokens(entity_id).await?;
        Ok(tokens > self.config.summary_threshold)
    }

    async fn estimate_tokens(&self, entity_id: &str) -> Result<usize, NarraError> {
        let (_, _, content) = match self.get_entity_full_content(entity_id).await? {
            Some(data) => data,
            None => return Ok(0),
        };

        Ok(Self::estimate_tokens_for_text(&content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(CachedSummaryService::estimate_tokens_for_text(""), 0);
    }

    #[test]
    fn test_estimate_tokens_short_text() {
        // "hello world" = 11 chars, ceil(11/4) = 3
        assert_eq!(
            CachedSummaryService::estimate_tokens_for_text("hello world"),
            3
        );
    }

    #[test]
    fn test_estimate_tokens_exact_multiple() {
        // 8 chars, ceil(8/4) = 2
        assert_eq!(
            CachedSummaryService::estimate_tokens_for_text("abcdefgh"),
            2
        );
    }

    #[test]
    fn test_compute_version_deterministic() {
        let v1 = CachedSummaryService::compute_version("same content");
        let v2 = CachedSummaryService::compute_version("same content");
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_compute_version_different_content() {
        let v1 = CachedSummaryService::compute_version("content a");
        let v2 = CachedSummaryService::compute_version("content b");
        assert_ne!(v1, v2);
    }

    /// Helper to create a CachedSummaryService with custom config for testing generate_summary.
    /// Uses a dummy DB that won't be called (we only test the pure generate_summary method).
    fn make_summary_service(summary_target: usize) -> CachedSummaryService {
        CachedSummaryService {
            entity_repo: Arc::new(SurrealEntityRepository::new(Arc::new(
                // We need a NarraDb but won't use it. Use a disconnected client.
                surrealdb::Surreal::<surrealdb::engine::any::Any>::init(),
            ))),
            summary_cache: Cache::builder().max_capacity(1).build(),
            config: SummaryConfig {
                summary_threshold: 200,
                summary_target,
                cache_ttl_secs: 60,
            },
        }
    }

    #[test]
    fn test_generate_summary_under_target_returns_unchanged() {
        let svc = make_summary_service(50); // target_chars = 50 * 4 = 200
        let short = "This is short text.";
        assert_eq!(svc.generate_summary(short), short);
    }

    #[test]
    fn test_generate_summary_truncates_at_sentence_boundary() {
        let svc = make_summary_service(10); // target_chars = 10 * 4 = 40
        let text = "First sentence here. Second sentence follows. Third sentence ends.";
        let result = svc.generate_summary(text);
        assert!(result.ends_with("..."));
        assert!(result.contains("First sentence here."));
    }

    #[test]
    fn test_generate_summary_truncates_at_word_boundary() {
        let svc = make_summary_service(5); // target_chars = 5 * 4 = 20
                                           // No ". " within first 20 chars, so falls back to word boundary
        let text = "oneword twoword threeword fourword fiveword";
        let result = svc.generate_summary(text);
        assert!(result.ends_with("..."));
        // Should not cut mid-word
        assert!(!result.contains("threewo"));
    }
}
