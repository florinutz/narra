use crate::db::connection::NarraDb;
use crate::NarraError;

/// Phase 1 schema: Foundation tables (character, location, event, relates_to, knowledge)
const SCHEMA_001: &str = include_str!("migrations/001_initial_schema.surql");

/// Phase 2 schema: Character psychology, asymmetric relationships, scenes
const SCHEMA_002: &str = include_str!("migrations/002_phase2_schema.surql");

/// Phase 3 schema: Knowledge states with certainty and provenance
const SCHEMA_003: &str = include_str!("migrations/003_phase3_schema.surql");

/// Phase 4 schema: Search infrastructure with FULLTEXT indexes
const SCHEMA_004: &str = include_str!("migrations/004_phase4_search.surql");

/// Phase 7 schema: Notes for freeform worldbuilding content
const SCHEMA_005: &str = include_str!("migrations/005_phase7_notes.surql");

/// Phase 11 schema: Universe facts for world rules and constraints
const SCHEMA_006: &str = include_str!("migrations/006_phase11_facts.surql");

/// Phase 16 schema: Embedding fields for semantic search
const SCHEMA_007: &str = include_str!("migrations/007_phase16_embeddings.surql");

/// Phase 17 schema: Referential integrity constraints
const SCHEMA_008: &str = include_str!("migrations/008_phase17_referential_integrity.surql");

/// HNSW vector indexes for semantic search
const SCHEMA_009: &str = include_str!("migrations/009_hnsw_vector_indexes.surql");

/// Knowledge embeddings: embedding fields + HNSW index for knowledge table
const SCHEMA_010: &str = include_str!("migrations/010_knowledge_embeddings.surql");

/// Arc snapshots: temporal embedding history for character arc tracking
const SCHEMA_011: &str = include_str!("migrations/011_arc_snapshots.surql");

/// Perspective embeddings on perceives edges + arc_snapshot type extension
const SCHEMA_012: &str = include_str!("migrations/012_perspective_embeddings.surql");

/// Embedding infrastructure: relates_to embeddings + composite_text storage
const SCHEMA_013: &str = include_str!("migrations/013_embedding_infra.surql");

/// Character profile: replace typed wounds/desires/contradictions with flexible profile HashMap
const SCHEMA_014: &str = include_str!("migrations/014_character_profile.surql");

/// Drop unused HNSW vector indexes (brute-force cosine is reliable + fast at narrative scale)
const SCHEMA_015: &str = include_str!("migrations/015_drop_hnsw_indexes.surql");

/// Embedding metadata: world_meta table for model/dimensions/provider tracking
const SCHEMA_016: &str = include_str!("migrations/016_embedding_metadata.surql");

/// Character facets: multi-vector embeddings for faceted search
const SCHEMA_017: &str = include_str!("migrations/017_character_facets.surql");

/// Phases: persisted narrative phase detection results + membership edges
const SCHEMA_018: &str = include_str!("migrations/018_phases.surql");

/// Annotations: generic ML model outputs cached per entity
const SCHEMA_019: &str = include_str!("migrations/019_annotations.surql");

/// Apply the database schema to an initialized database connection.
///
/// This executes all DEFINE statements in the schema files, creating tables,
/// fields, and indexes. Migrations are applied in order:
/// - 001: Foundation tables (character, location, event, relates_to, knowledge)
/// - 002: Phase 2 extensions (character psychology, perceives edge, scene, participates_in, involved_in)
/// - 003: Phase 3 extensions (knows edge for temporal knowledge with certainty/provenance)
/// - 004: Phase 4 extensions (FULLTEXT indexes for search with BM25 ranking)
/// - 005: Phase 7 extensions (notes for freeform worldbuilding content)
/// - 006: Phase 11 extensions (universe facts for world rules and constraints)
/// - 007: Phase 16 extensions (embedding fields for semantic search)
/// - 008: Phase 17 extensions (referential integrity with REFERENCE clauses)
/// - 009: HNSW vector indexes for semantic search
/// - 010: Knowledge embeddings (embedding fields + HNSW index for knowledge)
/// - 011: Arc snapshots (temporal embedding history for character arc tracking)
/// - 012: Perspective embeddings (per-observer view embeddings on perceives edges)
/// - 013: Embedding infrastructure (relates_to embeddings + composite_text storage)
/// - 014: Character profile (replace wounds/desires/contradictions with flexible profile)
/// - 015: Drop unused HNSW vector indexes (brute-force cosine at narrative scale)
/// - 016: Embedding metadata (world_meta table for model/dimensions/provider tracking)
/// - 017: Character facets (multi-vector embeddings for faceted search)
/// - 018: Phases (persisted narrative phase detection results + membership edges)
/// - 019: Annotations (generic ML model outputs cached per entity)
///
/// It's safe to call multiple times - SurrealDB will update existing definitions
/// rather than fail.
///
/// # Arguments
///
/// * `db` - An initialized database connection
///
/// # Returns
///
/// `Ok(())` if schema applied successfully, `Err(NarraError)` otherwise.
///
/// # Example
///
/// ```no_run
/// # use narra::db::{connection::{init_db, DbConfig}, schema::apply_schema};
/// # use std::path::Path;
/// # async fn example() -> Result<(), narra::NarraError> {
/// let config = DbConfig::Embedded { path: Some("./data/narra.db".into()) };
/// let db = init_db(&config, Path::new("./data")).await?;
/// apply_schema(&db).await?;
/// # Ok(())
/// # }
/// ```
pub async fn apply_schema(db: &NarraDb) -> Result<(), NarraError> {
    db.query(SCHEMA_001).await?;
    db.query(SCHEMA_002).await?;
    db.query(SCHEMA_003).await?;
    db.query(SCHEMA_004).await?;
    db.query(SCHEMA_005).await?;
    db.query(SCHEMA_006).await?;
    db.query(SCHEMA_007).await?;
    db.query(SCHEMA_008).await?;
    db.query(SCHEMA_009).await?;
    db.query(SCHEMA_010).await?;
    db.query(SCHEMA_011).await?;
    db.query(SCHEMA_012).await?;
    db.query(SCHEMA_013).await?;
    db.query(SCHEMA_014).await?;
    db.query(SCHEMA_015).await?;
    db.query(SCHEMA_016).await?;
    db.query(SCHEMA_017).await?;
    db.query(SCHEMA_018).await?;
    db.query(SCHEMA_019).await?;
    Ok(())
}
