use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Maximum allowed limit for result counts (prevents unbounded queries).
pub const MAX_LIMIT: usize = 500;

/// Maximum allowed depth for graph traversals (prevents runaway recursion).
pub const MAX_DEPTH: usize = 6;

/// Default token budget per tool response (configurable via NARRA_TOKEN_BUDGET env var).
pub const DEFAULT_TOKEN_BUDGET: usize = 2000;

/// Maximum allowed token budget (prevents unbounded response sizes).
pub const MAX_TOKEN_BUDGET: usize = 8000;

/// Detail level for entity queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DetailLevel {
    /// Brief overview (names, IDs only)
    #[default]
    Summary,
    /// Standard details (most fields)
    Standard,
    /// Complete entity data
    Full,
}

/// Graph format for traversal results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GraphFormat {
    /// List of entities (flat)
    #[default]
    Flat,
    /// Nested tree structure
    Tree,
}

/// Query operations (read-only).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum QueryRequest {
    /// Look up a specific entity by ID.
    Lookup {
        entity_id: String,
        #[serde(default)]
        detail_level: Option<DetailLevel>,
    },
    /// Search entities by text query.
    Search {
        query: String,
        #[serde(default)]
        entity_types: Option<Vec<String>>,
        #[serde(default)]
        limit: Option<usize>,
        #[serde(default)]
        cursor: Option<String>,
    },
    /// Traverse entity graph.
    GraphTraversal {
        entity_id: String,
        depth: usize,
        #[serde(default)]
        format: Option<GraphFormat>,
    },
    /// Query character knowledge at a point in time.
    Temporal {
        character_id: String,
        #[serde(default)]
        event_id: Option<String>,
        #[serde(default)]
        event_name: Option<String>,
    },
    /// Get overview of entity type.
    Overview {
        entity_type: String,
        #[serde(default)]
        limit: Option<usize>,
    },
    /// List notes, optionally filtered by attached entity.
    ListNotes {
        /// Filter to notes attached to this entity
        #[serde(default)]
        entity_id: Option<String>,
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Get a specific universe fact by ID.
    GetFact { fact_id: String },
    /// List universe facts with optional filters.
    ListFacts {
        /// Filter by category (physics_magic, social_cultural, technology, or custom)
        #[serde(default)]
        category: Option<String>,
        /// Filter by enforcement level (informational, warning, strict)
        #[serde(default)]
        enforcement_level: Option<String>,
        /// Full-text search query
        #[serde(default)]
        search: Option<String>,
        /// Filter to facts linked to this entity (e.g., "character:alice")
        #[serde(default)]
        entity_id: Option<String>,
        #[serde(default)]
        limit: Option<usize>,
        #[serde(default)]
        cursor: Option<String>,
    },
    /// Unified search across the world. Mode controls search strategy:
    /// - "semantic": pure vector similarity (requires embeddings)
    /// - "hybrid": keyword + semantic combined (graceful degradation)
    /// - "reranked": hybrid + cross-encoder re-ranking (best precision)
    UnifiedSearch {
        /// Natural language query (e.g., "characters who struggle with duty")
        query: String,
        /// "semantic", "hybrid", or "reranked" (default: "hybrid")
        #[serde(default = "default_search_mode")]
        mode: String,
        /// Filter by entity types (empty = character, location, event, scene)
        #[serde(default)]
        entity_types: Option<Vec<String>>,
        /// Maximum results (default: 10)
        #[serde(default)]
        limit: Option<usize>,
        /// Metadata filters for narrowing results
        #[serde(default)]
        filter: Option<SearchMetadataFilter>,
        /// Filter results to a specific narrative phase (auto-detected).
        /// Run detect_phases first to discover available phase IDs.
        #[serde(default)]
        phase: Option<usize>,
    },
    /// Find entities that reference a target entity.
    /// Discovers "what scenes mention Alice?" without explicit reverse edges.
    ReverseQuery {
        /// Target entity ID (e.g., "character:alice")
        entity_id: String,
        /// Filter by referencing entity types (empty = all types)
        /// For characters: scene, event, knowledge
        /// For locations: scene
        /// For events: scene
        #[serde(default)]
        referencing_types: Option<Vec<String>>,
        /// Maximum results (default: 20)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Find connection paths between two characters.
    /// Shows how characters are connected via relationships and shared events.
    ConnectionPath {
        /// Starting character ID (e.g., "character:alice")
        from_id: String,
        /// Target character ID (e.g., "character:bob")
        to_id: String,
        /// Maximum hops to search (default: 3)
        #[serde(default)]
        max_hops: Option<usize>,
        /// Include event co-participation as connections (default: true)
        #[serde(default)]
        include_events: Option<bool>,
    },
    /// Compute centrality metrics for the character relationship network.
    /// Reveals structural protagonists, bridge characters, and isolated nodes.
    CentralityMetrics {
        #[serde(default)]
        scope: Option<String>,
        #[serde(default)]
        metrics: Option<Vec<String>>,
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Trace how information could propagate through the character network.
    /// Follows directed relationship paths to find who could learn something.
    InfluencePropagation {
        from_character_id: String,
        #[serde(default)]
        knowledge_fact: Option<String>,
        #[serde(default)]
        max_depth: Option<usize>,
    },
    /// Generate dramatic irony report showing knowledge asymmetries.
    /// Finds what some characters know that others don't, with scene count since asymmetry.
    DramaticIronyReport {
        #[serde(default)]
        character_id: Option<String>,
        #[serde(default)]
        min_scene_threshold: Option<usize>,
    },
    /// Cross-field semantic search across entity types.
    /// Free-form queries like "characters whose desires conflict with Alice's wounds".
    SemanticJoin {
        query: String,
        #[serde(default)]
        entity_types: Option<Vec<String>>,
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Discover thematic clusters from entity embeddings.
    /// Groups semantically similar entities to reveal emergent story themes.
    ThematicClustering {
        #[serde(default)]
        entity_types: Option<Vec<String>>,
        #[serde(default)]
        num_themes: Option<usize>,
    },
    /// Search character knowledge by meaning.
    /// Finds knowledge entities semantically similar to the query.
    /// Optionally filtered to a specific character's knowledge.
    SemanticKnowledge {
        /// Natural language query (e.g., "the royal succession")
        query: String,
        /// Filter to one character's knowledge (e.g., "character:alice")
        #[serde(default)]
        character_id: Option<String>,
        /// Maximum results (default: 10)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Graph + Vector search: find connected entities that match a concept.
    /// Traverses the graph from a seed entity, then ranks candidates by
    /// semantic similarity to the query.
    SemanticGraphSearch {
        /// Seed entity (e.g., "character:alice")
        entity_id: String,
        /// Maximum graph hops (default: 2)
        #[serde(default)]
        max_hops: Option<usize>,
        /// Semantic query (e.g., "betrayal and deception")
        query: String,
        /// Filter result entity types (empty = all)
        #[serde(default)]
        entity_types: Option<Vec<String>>,
        /// Maximum results (default: 10)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// View embedding snapshot history for an entity.
    /// Shows how a character or knowledge entity has evolved over time.
    ArcHistory {
        /// Entity ID (e.g., "character:alice")
        entity_id: String,
        /// Optional facet filter for character snapshots (identity, psychology, social, narrative)
        #[serde(default)]
        facet: Option<String>,
        /// Maximum snapshots to return (default: 50)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Compare arc trajectories of two entities.
    /// Shows convergence/divergence and trajectory similarity.
    ArcComparison {
        /// First entity ID (e.g., "character:alice")
        entity_id_a: String,
        /// Second entity ID (e.g., "character:bob")
        entity_id_b: String,
        /// Window: "recent:N" for last N snapshots, or omit for all
        #[serde(default)]
        window: Option<String>,
    },
    /// Rank entities by total embedding drift.
    /// Reveals which characters have changed the most.
    ArcDrift {
        /// Filter to "character" or "knowledge" (default: both)
        #[serde(default)]
        entity_type: Option<String>,
        /// Maximum results (default: 20)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Get entity state at a specific moment (nearest snapshot to event).
    ArcMoment {
        /// Entity ID (e.g., "character:alice")
        entity_id: String,
        /// Event ID to find snapshot near (omit for latest)
        #[serde(default)]
        event_id: Option<String>,
    },
    /// Search perspective embeddings semantically.
    /// "Who does Bob see as threatening?" or "Find all perspectives about betrayal."
    PerspectiveSearch {
        /// Natural language query (e.g., "threatening and dangerous")
        query: String,
        /// Filter to perspectives FROM this observer (e.g., "character:bob")
        #[serde(default)]
        observer_id: Option<String>,
        /// Filter to perspectives OF this target (e.g., "character:alice")
        #[serde(default)]
        target_id: Option<String>,
        /// Maximum results (default: 10)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Measure how wrong an observer is about a target.
    /// Cosine distance between "observer's target" and "the real target."
    PerceptionGap {
        /// Observer character (e.g., "character:bob")
        observer_id: String,
        /// Target character (e.g., "character:alice")
        target_id: String,
    },
    /// How do multiple observers see the same target?
    /// Returns per-observer gap + pairwise agreement between observers.
    PerceptionMatrix {
        /// Target character (e.g., "character:alice")
        target_id: String,
        /// Maximum observers to include (default: 20)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// How has an observer's view of a target changed over time?
    /// Uses arc snapshots to track perspective evolution + gap convergence.
    PerceptionShift {
        /// Observer character (e.g., "character:bob")
        observer_id: String,
        /// Target character (e.g., "character:alice")
        target_id: String,
    },
    /// Surface character pairs with high perception asymmetry and few shared scenes.
    /// Finds "scenes waiting to be written" — unresolved tensions in the narrative.
    UnresolvedTensions {
        /// Maximum results (default: 10)
        #[serde(default)]
        limit: Option<usize>,
        /// Minimum asymmetry threshold (default: 0.1)
        #[serde(default)]
        min_asymmetry: Option<f32>,
        /// Maximum shared scenes to qualify (default: no limit)
        #[serde(default)]
        max_shared_scenes: Option<usize>,
    },
    /// Find thematic clusters missing expected entity types.
    /// Reveals "themes without protagonists" or "characters without plot."
    ThematicGaps {
        /// Minimum cluster size to analyze (default: 3)
        #[serde(default)]
        min_cluster_size: Option<usize>,
        /// Expected entity types per cluster (default: ["character", "event"])
        #[serde(default)]
        expected_types: Option<Vec<String>>,
    },
    /// Find relationships similar to a reference relationship.
    /// Searches both perceives and relates_to edges by embedding similarity.
    SimilarRelationships {
        /// Observer/from character in the reference relationship
        observer_id: String,
        /// Target/to character in the reference relationship
        target_id: String,
        /// Edge type to search from: "perceives" or "relates_to" (default: auto-detect)
        #[serde(default)]
        edge_type: Option<String>,
        /// Optional text to bias the search direction (e.g., "with more tension")
        #[serde(default)]
        bias: Option<String>,
        /// Maximum results (default: 10)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Find knowledge conflicts where characters believe contradictory facts.
    /// Shows what each character believes wrongly, with truth values and certainty.
    KnowledgeConflicts {
        /// Filter to a specific character (e.g., "character:alice" or "alice")
        #[serde(default)]
        character_id: Option<String>,
        /// Maximum results (default: 50)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Diagnostic: embedding coverage and staleness report.
    EmbeddingHealth,
    /// Preview embedding shift from a character learning a fact — without committing.
    /// "What if Alice learned Bob's secret?" Shows impact, conflicts, and cascade.
    WhatIf {
        /// Character who would learn the fact (e.g., "alice" or "character:alice")
        character_id: String,
        /// Knowledge fact ID (e.g., "secret1" or "knowledge:secret1")
        fact_id: String,
        /// Certainty level (default: "knows")
        #[serde(default)]
        certainty: Option<String>,
        /// Who told them (optional, for composite context)
        #[serde(default)]
        source_character: Option<String>,
    },
    /// Validate a single entity for all consistency issues (fact violations, timeline, relationships).
    ValidateEntity {
        /// Full entity ID (e.g., "character:alice", "scene:chapter1")
        entity_id: String,
    },
    /// Investigate what contradicts a specific entity or fact via graph traversal.
    InvestigateContradictions {
        /// Full entity ID or fact ID to investigate
        entity_id: String,
        /// Maximum depth for graph traversal (default: 3)
        #[serde(default = "default_investigate_depth")]
        max_depth: usize,
    },
    /// Detect knowledge asymmetries between two specific characters.
    /// Shows what each knows that the other doesn't, enriched with tension and enforcement context.
    KnowledgeAsymmetries {
        /// First character ID (e.g., "character:alice" or just "alice")
        character_a: String,
        /// Second character ID (e.g., "character:bob" or just "bob")
        character_b: String,
    },
    /// High-level narrative situation report: top irony, conflicts, tensions, themes.
    /// Combines multiple analytics into a single strategic overview.
    SituationReport {
        /// Detail level: "summary", "standard" (default), or "full"
        #[serde(default)]
        detail_level: Option<String>,
    },
    /// Comprehensive dossier for a single character: network position, knowledge,
    /// influence, perceptions, and narrative suggestions.
    CharacterDossier {
        /// Character ID (e.g., "character:alice" or just "alice")
        character_id: String,
        /// Detail level: "summary", "standard" (default), or "full"
        #[serde(default)]
        detail_level: Option<String>,
    },
    /// Scene planning for a set of characters about to meet.
    /// Shows pairwise dynamics, irony opportunities, tensions, and applicable facts.
    ScenePlanning {
        /// Character IDs (e.g., ["character:alice", "character:bob"])
        character_ids: Vec<String>,
        /// Detail level: "summary", "standard" (default), or "full"
        #[serde(default)]
        detail_level: Option<String>,
    },
    /// Compute growth vector for an entity: where is it heading based on arc snapshots.
    /// Finds entities aligned with the trajectory extrapolation.
    GrowthVector {
        /// Entity ID (e.g., "character:alice")
        entity_id: String,
        /// Maximum trajectory neighbors (default: 5)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Compute misperception vector: what does the observer get wrong about the target?
    /// Shows the direction and magnitude of misperception.
    MisperceptionVector {
        /// Observer character ID (e.g., "character:bob")
        observer_id: String,
        /// Target character ID (e.g., "character:alice")
        target_id: String,
        /// Maximum neighbors to show (default: 5)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Analyze convergence/divergence between two entities over time.
    /// Uses arc snapshots to compute per-snapshot similarity trend.
    ConvergenceAnalysis {
        /// First entity ID (e.g., "character:alice")
        entity_a: String,
        /// Second entity ID (e.g., "character:bob")
        entity_b: String,
        /// Maximum snapshots to analyze (default: 50)
        #[serde(default)]
        window: Option<usize>,
    },
    /// Find entities at the semantic midpoint of two entities.
    /// "What bridges betrayal and redemption?"
    SemanticMidpoint {
        /// First entity ID
        entity_a: String,
        /// Second entity ID
        entity_b: String,
        /// Maximum results (default: 5)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Preview change impact before mutating. Returns affected entities by severity.
    AnalyzeImpact {
        /// The entity ID to analyze impact for
        entity_id: String,
        /// Optional: Proposed change description for more accurate analysis
        #[serde(default)]
        proposed_change: Option<String>,
        /// Whether to include full details of affected entities (default: false)
        #[serde(default)]
        include_details: Option<bool>,
    },
    /// Tension matrix: all character pairs with tension scores above a threshold.
    /// Shows the emotional landscape of the narrative world.
    TensionMatrix {
        /// Minimum tension level to include (default: 1)
        #[serde(default)]
        min_tension: Option<i32>,
        /// Maximum pairs to return (default: 50)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Knowledge gap analysis: important unknowns, blind spots, and false beliefs
    /// for a specific character.
    KnowledgeGapAnalysis {
        /// Character ID (e.g., "character:alice")
        character_id: String,
    },
    /// Relationship strength map: weighted graph of relationship intensities
    /// for a character's connections.
    RelationshipStrengthMap {
        /// Character ID (e.g., "character:alice")
        character_id: String,
        /// Maximum relationships to return (default: 20)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Narrative threads: open plot lines, unresolved knowledge gaps, stale arcs.
    NarrativeThreads {
        /// Filter by status: "open", "stale", "all" (default: "all")
        #[serde(default)]
        status: Option<String>,
        /// Maximum threads to return (default: 20)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Character voice profile: speech tendencies, certainty patterns,
    /// emotional register, knowledge-informed vocabulary guidance.
    CharacterVoice {
        /// Character ID (e.g., "character:alice")
        character_id: String,
    },
    /// Search character facet embeddings.
    /// Returns characters ranked by semantic similarity on a specific facet.
    FacetedSearch {
        /// Natural language query (e.g., "inner conflict")
        query: String,
        /// Facet name: "identity", "psychology", "social", or "narrative"
        facet: String,
        /// Maximum results (default: 10)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Weighted multi-facet search across character dimensions.
    /// Example: `{"psychology": 0.6, "social": 0.4}` searches 60% psychology, 40% social.
    MultiFacetSearch {
        /// Natural language query
        query: String,
        /// Facet weights (e.g., {"psychology": 0.6, "social": 0.4})
        /// Weights are normalized to sum to 1.0
        weights: HashMap<String, f32>,
        /// Maximum results (default: 10)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Diagnostic: view all facet statuses and inter-facet similarities for a character.
    CharacterFacets {
        /// Character ID (e.g., "character:alice")
        character_id: String,
    },
    /// Auto-detect narrative phases by clustering entities using composite
    /// narrative distance (embeddings + event sequence + scene co-occurrence).
    DetectPhases {
        /// Entity types to include (default: all embeddable)
        #[serde(default)]
        entity_types: Option<Vec<String>>,
        /// Number of phases (auto-detected if not specified)
        #[serde(default)]
        num_phases: Option<usize>,
        /// Content weight 0.0-1.0 (default: 0.6)
        #[serde(default)]
        content_weight: Option<f32>,
        /// Neighborhood weight 0.0-1.0 (default: 0.25)
        #[serde(default)]
        neighborhood_weight: Option<f32>,
        /// Temporal weight 0.0-1.0 (default: 0.15)
        #[serde(default)]
        temporal_weight: Option<f32>,
        /// Persist detected phases to database for instant loading
        #[serde(default)]
        save: Option<bool>,
    },
    /// Load previously saved phases from database (returns nothing if none saved).
    LoadPhases,
    /// Find entities narratively close to an anchor entity.
    /// Uses composite narrative distance (embedding similarity +
    /// scene co-occurrence + event sequence proximity).
    QueryAround {
        /// Anchor entity ID (e.g., "event:betrayal", "character:alice")
        anchor_id: String,
        /// Entity types to include in results (default: all embeddable)
        #[serde(default)]
        entity_types: Option<Vec<String>>,
        /// Maximum results (default: 20)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Classify the emotional tone of an entity's composite text.
    /// Returns emotion labels with confidence scores (GoEmotions, 28 labels).
    /// Uses cached annotations — results are recomputed only when the entity changes.
    Emotions {
        /// Entity ID (e.g., "character:alice", "event:betrayal")
        entity_id: String,
    },
    /// Classify narrative themes present in an entity's text using zero-shot NLI.
    /// Returns theme labels with entailment scores. Default themes are narrative
    /// archetypes (love, betrayal, power, etc.). Custom themes can be supplied.
    /// Uses cached annotations for default themes.
    Themes {
        /// Entity ID (e.g., "character:alice", "scene:confrontation")
        entity_id: String,
        /// Optional custom themes to classify against. If omitted, uses default
        /// narrative themes (love, betrayal, revenge, identity, power, etc.)
        #[serde(default)]
        themes: Option<Vec<String>>,
    },
    /// Extract named entities (persons, locations, organizations) from an entity's text.
    /// Uses BERT-based NER with BIO tagging. Results are cached as annotations.
    ExtractEntities {
        /// Entity ID (e.g., "scene:confrontation", "event:betrayal")
        entity_id: String,
    },
    /// Detect narrative tensions between characters using profile analysis,
    /// knowledge contradictions, and conflicting loyalties.
    /// Goes beyond the perceives-based tension_matrix to find structural tensions.
    NarrativeTensions {
        /// Maximum tensions to return (default: 20)
        #[serde(default)]
        limit: Option<usize>,
        /// Minimum severity threshold 0.0–1.0 (default: 0.0)
        #[serde(default)]
        min_severity: Option<f32>,
    },
    /// Infer narrative roles for characters based on graph topology,
    /// knowledge patterns, relationship types, and profile traits.
    /// Returns richer roles than basic centrality: mentor, enigma, deceived,
    /// keeper_of_secrets, antagonist, bridge, information_broker, etc.
    InferRoles {
        /// Maximum characters to return (default: 20)
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Identify entities that bridge multiple narrative phases.
    /// These "transition points" are characters, events, or locations
    /// that connect different arcs in the story.
    DetectTransitions {
        /// Entity types to include (default: all embeddable)
        #[serde(default)]
        entity_types: Option<Vec<String>>,
        /// Number of phases (auto-detected if not specified)
        #[serde(default)]
        num_phases: Option<usize>,
        /// Content weight 0.0–1.0 (default: 0.6)
        #[serde(default)]
        content_weight: Option<f32>,
        /// Neighborhood weight 0.0–1.0 (default: 0.25)
        #[serde(default)]
        neighborhood_weight: Option<f32>,
        /// Temporal weight 0.0–1.0 (default: 0.15)
        #[serde(default)]
        temporal_weight: Option<f32>,
    },
}

fn default_investigate_depth() -> usize {
    3
}

fn default_search_mode() -> String {
    "hybrid".into()
}

/// A single issue in validation results.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ValidationIssue {
    pub issue_type: String,
    pub severity: String,
    pub message: String,
    pub suggested_fix: Option<String>,
    pub confidence: f32,
}

/// Typed metadata filter for semantic and hybrid search.
/// Provides JSON Schema so LLMs can discover available filter fields.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct SearchMetadataFilter {
    /// Filter characters by role (e.g. "warrior"). Checks array membership.
    #[serde(default)]
    pub roles: Option<String>,
    /// Filter by name (case-insensitive substring match)
    #[serde(default)]
    pub name: Option<String>,
    /// Filter events with sequence >= this value
    #[serde(default)]
    pub sequence_min: Option<i64>,
    /// Filter events with sequence <= this value
    #[serde(default)]
    pub sequence_max: Option<i64>,
    /// Filter locations by type (e.g. "castle", "forest")
    #[serde(default)]
    pub loc_type: Option<String>,
}

/// Spec for a character in batch creation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CharacterSpec {
    /// Caller-specified ID slug (e.g. "alice"). Auto-generated if omitted.
    #[serde(default)]
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub aliases: Option<Vec<String>>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub profile: Option<HashMap<String, Vec<String>>>,
}

/// Spec for a location in batch creation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LocationSpec {
    /// Caller-specified ID slug (e.g. "castle"). Auto-generated if omitted.
    #[serde(default)]
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// Parent location ID (e.g. "location:castle")
    #[serde(default)]
    pub parent_id: Option<String>,
    /// Location type (defaults to "place" if omitted)
    #[serde(default)]
    pub loc_type: Option<String>,
}

/// Spec for an event in batch creation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EventSpec {
    /// Caller-specified ID slug (e.g. "betrayal"). Auto-generated if omitted.
    #[serde(default)]
    pub id: Option<String>,
    pub title: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub sequence: Option<i32>,
    #[serde(default)]
    pub date: Option<String>,
    #[serde(default)]
    pub date_precision: Option<String>,
}

/// Spec for a relationship in batch creation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RelationshipSpec {
    pub from_character_id: String,
    pub to_character_id: String,
    pub rel_type: String,
    #[serde(default)]
    pub subtype: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
}

/// Conflict resolution mode for import operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ConflictMode {
    /// Return error for duplicate IDs (continue with remaining entities)
    #[default]
    Error,
    /// Silently skip duplicate IDs
    Skip,
    /// Update existing entities with imported fields
    Update,
}

/// Spec for a scene participant in import.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ParticipantSpec {
    pub character_id: String,
    #[serde(default = "default_participant_role")]
    pub role: String,
    #[serde(default)]
    pub notes: Option<String>,
}

fn default_participant_role() -> String {
    "supporting".to_string()
}

/// Spec for a scene in import.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SceneSpec {
    #[serde(default)]
    pub id: Option<String>,
    pub title: String,
    pub event_id: String,
    pub location_id: String,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub secondary_locations: Vec<String>,
    #[serde(default)]
    pub participants: Vec<ParticipantSpec>,
}

/// Spec for a knowledge entry in import.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct KnowledgeSpec {
    pub character_id: String,
    pub target_id: String,
    pub fact: String,
    #[serde(default = "default_certainty")]
    pub certainty: String,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub source_character_id: Option<String>,
    #[serde(default)]
    pub event_id: Option<String>,
}

fn default_certainty() -> String {
    "knows".to_string()
}

/// Spec for a fact-to-entity link in import.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FactLinkSpec {
    pub entity_id: String,
    #[serde(default = "default_link_type")]
    pub link_type: String,
    #[serde(default)]
    pub confidence: Option<f32>,
}

fn default_link_type() -> String {
    "manual".to_string()
}

/// Spec for a universe fact in import.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FactSpec {
    #[serde(default)]
    pub id: Option<String>,
    pub title: String,
    pub description: String,
    #[serde(default)]
    pub categories: Vec<String>,
    #[serde(default)]
    pub enforcement_level: Option<String>,
    #[serde(default)]
    pub applies_to: Vec<FactLinkSpec>,
}

/// Spec for a note in import.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct NoteSpec {
    #[serde(default)]
    pub id: Option<String>,
    pub title: String,
    pub body: String,
    #[serde(default)]
    pub attach_to: Vec<String>,
}

/// Full YAML import document for bootstrapping a story world.
///
/// Read `narra://schema/import-template` for a commented YAML template with examples,
/// or `narra://schema/import-schema` for the JSON Schema.
///
/// Entity types are processed in dependency order:
/// characters → locations → events → scenes → relationships → knowledge → notes → facts.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct NarraImport {
    #[serde(default)]
    pub characters: Vec<CharacterSpec>,
    #[serde(default)]
    pub locations: Vec<LocationSpec>,
    #[serde(default)]
    pub events: Vec<EventSpec>,
    #[serde(default)]
    pub scenes: Vec<SceneSpec>,
    #[serde(default)]
    pub relationships: Vec<RelationshipSpec>,
    #[serde(default)]
    pub knowledge: Vec<KnowledgeSpec>,
    #[serde(default)]
    pub notes: Vec<NoteSpec>,
    #[serde(default)]
    pub facts: Vec<FactSpec>,
}

/// Per-entity-type import result.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ImportTypeResult {
    pub entity_type: String,
    pub created: usize,
    pub skipped: usize,
    pub updated: usize,
    pub errors: Vec<String>,
}

/// Aggregate import result.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ImportResult {
    pub total_created: usize,
    pub total_skipped: usize,
    pub total_updated: usize,
    pub total_errors: usize,
    pub by_type: Vec<ImportTypeResult>,
}

/// Mutation operations (write).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum MutationRequest {
    /// Create a new character.
    CreateCharacter {
        /// Caller-specified ID slug (e.g. "alice"). Auto-generated if omitted.
        #[serde(default)]
        id: Option<String>,
        name: String,
        #[serde(default)]
        role: Option<String>,
        #[serde(default)]
        aliases: Option<Vec<String>>,
        #[serde(default)]
        description: Option<String>,
        /// Flexible profile: keys are categories (e.g. "wound", "secret"),
        /// values are lists of entries.
        #[serde(default)]
        profile: Option<HashMap<String, Vec<String>>>,
    },
    /// Create a new location.
    CreateLocation {
        /// Caller-specified ID slug (e.g. "castle"). Auto-generated if omitted.
        #[serde(default)]
        id: Option<String>,
        name: String,
        #[serde(default)]
        parent_id: Option<String>,
        #[serde(default)]
        description: Option<String>,
    },
    /// Create a new event.
    CreateEvent {
        /// Caller-specified ID slug (e.g. "betrayal"). Auto-generated if omitted.
        #[serde(default)]
        id: Option<String>,
        title: String,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        sequence: Option<i32>,
        #[serde(default)]
        date: Option<String>,
        #[serde(default)]
        date_precision: Option<String>,
    },
    /// Create a new scene.
    CreateScene {
        title: String,
        event_id: String,
        location_id: String,
        #[serde(default)]
        summary: Option<String>,
    },
    /// Update an entity's fields.
    Update {
        entity_id: String,
        fields: serde_json::Value,
    },
    /// Record character knowledge.
    RecordKnowledge {
        character_id: String,
        target_id: String,
        fact: String,
        certainty: String,
        #[serde(default)]
        method: Option<String>,
        #[serde(default)]
        source_character_id: Option<String>,
        #[serde(default)]
        event_id: Option<String>,
    },
    /// Delete an entity.
    Delete {
        entity_id: String,
        #[serde(default)]
        hard: Option<bool>,
    },
    /// Create a new note.
    CreateNote {
        title: String,
        body: String,
        /// Optional entity IDs to attach the note to
        #[serde(default)]
        attach_to: Option<Vec<String>>,
    },
    /// Attach an existing note to an entity.
    AttachNote { note_id: String, entity_id: String },
    /// Detach a note from an entity.
    DetachNote { note_id: String, entity_id: String },
    /// Create a new universe fact.
    CreateFact {
        title: String,
        description: String,
        /// Categories (physics_magic, social_cultural, technology, or custom string)
        #[serde(default)]
        categories: Option<Vec<String>>,
        /// Enforcement level: informational, warning (default), strict
        #[serde(default)]
        enforcement_level: Option<String>,
    },
    /// Update an existing universe fact.
    UpdateFact {
        fact_id: String,
        #[serde(default)]
        title: Option<String>,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        categories: Option<Vec<String>>,
        #[serde(default)]
        enforcement_level: Option<String>,
    },
    /// Delete a universe fact.
    DeleteFact { fact_id: String },
    /// Link a universe fact to an entity.
    LinkFact { fact_id: String, entity_id: String },
    /// Unlink a universe fact from an entity.
    UnlinkFact { fact_id: String, entity_id: String },
    /// Create a relationship between two characters.
    CreateRelationship {
        from_character_id: String,
        to_character_id: String,
        rel_type: String,
        #[serde(default)]
        subtype: Option<String>,
        #[serde(default)]
        label: Option<String>,
    },
    /// Batch-create multiple characters in one call.
    BatchCreateCharacters { characters: Vec<CharacterSpec> },
    /// Batch-create multiple locations in one call.
    BatchCreateLocations { locations: Vec<LocationSpec> },
    /// Batch-create multiple events in one call.
    BatchCreateEvents { events: Vec<EventSpec> },
    /// Batch-create multiple relationships in one call.
    BatchCreateRelationships {
        relationships: Vec<RelationshipSpec>,
    },
    /// Batch-record knowledge for multiple characters in one call.
    /// Each entry creates a knowledge entity and a knowledge state edge.
    BatchRecordKnowledge { knowledge: Vec<KnowledgeSpec> },
    /// Generate embeddings for all existing entities.
    /// Run after first setup or when embedding model changes.
    BackfillEmbeddings {
        /// Optional: only backfill specific entity type
        #[serde(default)]
        entity_type: Option<String>,
    },
    /// Create baseline arc snapshots for entities with embeddings but no snapshots yet.
    /// Run once after deploying arc tracking to capture initial state.
    BaselineArcSnapshots {
        /// Filter to "character" or "knowledge" (default: both)
        #[serde(default)]
        entity_type: Option<String>,
    },
    /// Mark entity as protected. Protected entities trigger CRITICAL severity in impact analysis.
    ProtectEntity { entity_id: String },
    /// Remove protection from entity, restoring normal severity calculations.
    UnprotectEntity { entity_id: String },
    /// Import a full world from a structured document. Processes entities in dependency order:
    /// characters → locations → events → scenes → relationships → knowledge → notes → facts.
    ///
    /// ID conventions: use short lowercase slugs (e.g. "alice", "castle"). IDs become
    /// SurrealDB record keys (character:alice). Omit id for auto-generation.
    ///
    /// Cross-references use full record format: event_id="event:arrival",
    /// location_id="location:castle", character_id="character:alice".
    ///
    /// Locations: parents must appear before children. Knowledge: always appended (no conflict check).
    ///
    /// A complete YAML template with examples is available via the narra://schema/import-template
    /// resource (human users can reference it with @narra:narra://schema/import-template).
    ImportYaml {
        /// The import document with entity arrays: characters, locations, events, scenes,
        /// relationships, knowledge, notes, facts
        import: NarraImport,
        /// How to handle duplicate IDs: error (default) reports and continues,
        /// skip silently ignores, update merges fields into existing entities
        #[serde(default)]
        on_conflict: ConflictMode,
    },
    /// Detect and persist narrative phases to database. Re-detection replaces all saved phases.
    SavePhases {
        /// Entity types to include (default: all embeddable)
        #[serde(default)]
        entity_types: Option<Vec<String>>,
        /// Number of phases (auto-detected if not specified)
        #[serde(default)]
        num_phases: Option<usize>,
        /// Content weight 0.0-1.0 (default: 0.6)
        #[serde(default)]
        content_weight: Option<f32>,
        /// Neighborhood weight 0.0-1.0 (default: 0.25)
        #[serde(default)]
        neighborhood_weight: Option<f32>,
        /// Temporal weight 0.0-1.0 (default: 0.15)
        #[serde(default)]
        temporal_weight: Option<f32>,
    },
    /// Clear all saved phases from database.
    ClearPhases,
    /// Run ML annotation pipeline on all entities of the specified types.
    /// Annotates entities with emotion, theme, and NER classifiers in parallel.
    /// Results are cached as annotations in the database.
    AnnotateEntities {
        /// Entity types to annotate (e.g. ["character", "event", "scene"]).
        /// Defaults to ["character", "event", "scene"] if omitted.
        #[serde(default)]
        entity_types: Option<Vec<String>>,
        /// Whether to run emotion classification (default: true).
        #[serde(default = "default_true")]
        run_emotions: bool,
        /// Whether to run theme classification (default: true).
        #[serde(default = "default_true")]
        run_themes: bool,
        /// Whether to run named entity recognition (default: true).
        #[serde(default = "default_true")]
        run_ner: bool,
        /// Max concurrency for parallel processing (default: 4).
        #[serde(default)]
        concurrency: Option<usize>,
    },
}

fn default_true() -> bool {
    true
}

/// Single entity result.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EntityResult {
    /// Entity ID (e.g., "character:alice")
    pub id: String,
    /// Entity type
    pub entity_type: String,
    /// Display name/title
    pub name: String,
    /// Entity content (formatted text)
    pub content: String,
    /// Relevance/confidence score (0.0-1.0)
    #[serde(default)]
    pub confidence: Option<f32>,
    /// Last modification timestamp
    #[serde(default)]
    pub last_modified: Option<String>,
}

// =============================================================================
// Dedicated Tool Input Structs (typed JSON Schema for MCP)
// Each struct becomes a dedicated MCP tool with full JSON Schema validation.
// =============================================================================

// --- Essential Tool Inputs ---

/// Input for semantic_search tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SemanticSearchInput {
    /// Natural language query (e.g., "characters who struggle with duty")
    pub query: String,
    /// Filter by entity types: character, location, event, scene, knowledge
    #[serde(default)]
    pub entity_types: Option<Vec<String>>,
    /// Maximum results (default: 10)
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Input for dossier tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DossierInput {
    /// Character ID (e.g., "character:alice" or just "alice")
    pub character_id: String,
}

/// Input for scene_prep tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ScenePrepInput {
    /// Character IDs for the scene (at least 2, e.g., ["character:alice", "character:bob"])
    pub character_ids: Vec<String>,
}

/// Input for overview tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OverviewInput {
    /// Entity type: "character", "location", "event", "scene", or "all" (default: "all")
    #[serde(default = "default_overview_type")]
    pub entity_type: String,
    /// Maximum results per type (default: 20)
    #[serde(default)]
    pub limit: Option<usize>,
}

fn default_overview_type() -> String {
    "all".to_string()
}

/// Input for record_knowledge tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RecordKnowledgeInput {
    /// Character who learns (e.g., "character:alice" or "alice")
    pub character_id: String,
    /// What/who the knowledge is about (e.g., "character:bob")
    pub target_id: String,
    /// The fact being learned (natural language)
    pub fact: String,
    /// Certainty level: knows, suspects, believes_wrongly, uncertain
    pub certainty: String,
    /// How they learned it (e.g., "witnessed", "told_by", "overheard")
    #[serde(default)]
    pub method: Option<String>,
    /// Who told them (if learned from another character)
    #[serde(default)]
    pub source_character_id: Option<String>,
    /// Event where knowledge was gained
    #[serde(default)]
    pub event_id: Option<String>,
}

// --- Standard Tool Inputs ---

/// Input for search tool (keyword).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct KeywordSearchInput {
    /// Search query (names, titles, keywords)
    pub query: String,
    /// Filter by entity types: character, location, event, scene
    #[serde(default)]
    pub entity_types: Option<Vec<String>>,
    /// Maximum results (default: 10)
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Input for lookup tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LookupInput {
    /// Entity ID (e.g., "character:alice", "event:betrayal")
    pub entity_id: String,
    /// Detail level: "summary" (default), "standard", or "full"
    #[serde(default)]
    pub detail_level: Option<String>,
}

/// Input for create_character tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CreateCharacterInput {
    /// Character name (display name)
    pub name: String,
    /// ID slug (e.g., "alice"). Auto-generated if omitted.
    #[serde(default)]
    pub id: Option<String>,
    /// Role in the story (e.g., "protagonist", "antagonist")
    #[serde(default)]
    pub role: Option<String>,
    /// Alternative names or titles
    #[serde(default)]
    pub aliases: Option<Vec<String>>,
    /// Character description
    #[serde(default)]
    pub description: Option<String>,
    /// Profile: keys are categories (wound, secret, desire, contradiction), values are entry lists
    #[serde(default)]
    pub profile: Option<HashMap<String, Vec<String>>>,
}

/// Input for create_relationship tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CreateRelationshipInput {
    /// Source character ID (e.g., "character:alice")
    pub from_character_id: String,
    /// Target character ID (e.g., "character:bob")
    pub to_character_id: String,
    /// Relationship type (e.g., "ally", "enemy", "mentor", "rival")
    pub rel_type: String,
    /// Relationship subtype for finer classification
    #[serde(default)]
    pub subtype: Option<String>,
    /// Human-readable label (e.g., "childhood friends")
    #[serde(default)]
    pub label: Option<String>,
}

/// Input for irony_report tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IronyReportInput {
    /// Focus on specific character (omit for full world report)
    #[serde(default)]
    pub character_id: Option<String>,
    /// Minimum scene count since asymmetry to include (default: 3)
    #[serde(default)]
    pub min_scene_threshold: Option<usize>,
}

/// Input for update_entity tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UpdateEntityInput {
    /// Entity ID (e.g., "character:alice", "event:betrayal")
    pub entity_id: String,
    /// Fields to update (JSON object with field names and new values)
    pub fields: serde_json::Value,
}

/// Input for knowledge_asymmetries tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct KnowledgeAsymmetriesInput {
    /// First character ID (e.g., "character:alice" or "alice")
    pub character_a: String,
    /// Second character ID (e.g., "character:bob" or "bob")
    pub character_b: String,
}

/// Input for validate_entity tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ValidateEntityInput {
    /// Entity ID to validate (e.g., "character:alice", "scene:chapter1")
    pub entity_id: String,
}

// =============================================================================
// MCP Tool Input Wrappers
// Free-form parameter pattern for MCP compatibility.
// Top-level schema is type: "object", params are validated at runtime.
// =============================================================================

/// Free-form input for query tool (runtime deserialization to QueryRequest).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryInput {
    /// Operation name (lookup, search, semantic_search, etc.)
    pub operation: String,
    /// Optional per-request token budget. Overrides tool-type default and env var.
    /// Capped at MAX_TOKEN_BUDGET (8000).
    #[serde(default)]
    pub token_budget: Option<usize>,
    /// Operation-specific parameters (validated at runtime)
    #[serde(flatten)]
    pub params: serde_json::Map<String, serde_json::Value>,
}

/// Free-form input for mutate tool (runtime deserialization to MutationRequest).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MutationInput {
    /// Operation name (create_character, update, delete, etc.)
    pub operation: String,
    /// Operation-specific parameters (validated at runtime)
    #[serde(flatten)]
    pub params: serde_json::Map<String, serde_json::Value>,
}

/// Free-form input for session tool (runtime deserialization to SessionRequest).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SessionInput {
    /// Operation name (get_context, pin_entity, unpin_entity)
    pub operation: String,
    /// Operation-specific parameters (validated at runtime)
    #[serde(flatten)]
    pub params: serde_json::Map<String, serde_json::Value>,
}

/// Information about response truncation due to token budget constraints.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TruncationInfo {
    /// Reason for truncation ("token_budget" or "result_limit")
    pub reason: String,
    /// Total results before truncation
    pub original_count: usize,
    /// Results actually returned
    pub returned_count: usize,
    /// Hint for getting more data (e.g., "Use limit=50" or "Narrow your query")
    pub suggestion: String,
}

/// Query response with multiple results.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryResponse {
    /// Result entities
    pub results: Vec<EntityResult>,
    /// Total count (may exceed results.len() if paginated)
    pub total: usize,
    /// Pagination cursor for next page
    #[serde(default)]
    pub next_cursor: Option<String>,
    /// Helpful hints for the user
    #[serde(default)]
    pub hints: Vec<String>,
    /// Estimated token count for this response
    pub token_estimate: usize,
    /// Truncation info if response was truncated due to token budget
    #[serde(default)]
    pub truncated: Option<TruncationInfo>,
}

/// Impact summary for mutations.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ImpactSummary {
    /// Number of entities affected
    pub affected_count: usize,
    /// Severity level (low, medium, high, critical)
    pub severity: String,
    /// Warnings about protected entities or conflicts
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Mutation response.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MutationResponse {
    /// The created/modified entity (summary for batch operations)
    pub entity: EntityResult,
    /// Individual results for batch operations
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entities: Option<Vec<EntityResult>>,
    /// Impact analysis (if applicable)
    #[serde(default)]
    pub impact: Option<ImpactSummary>,
    /// Helpful hints for next steps
    #[serde(default)]
    pub hints: Vec<String>,
}

/// Disambiguation result when entity is ambiguous.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DisambiguationResult {
    /// Best matching entity (if any)
    #[serde(default)]
    pub best_match: Option<EntityResult>,
    /// Confidence in best match (0.0-1.0)
    pub confidence: f32,
    /// Alternative matches to consider
    #[serde(default)]
    pub alternatives: Vec<EntityResult>,
}

/// Session management operations.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum SessionRequest {
    /// Get session summary: recent work, hot entities, pinned items, pending decisions.
    GetContext {
        /// Force full verbosity regardless of time elapsed
        #[serde(default)]
        force_full: bool,
    },
    /// Pin entity to working context. Pinned entities persist across sessions.
    PinEntity {
        /// Entity ID to pin (e.g., "character:alice")
        entity_id: String,
    },
    /// Unpin entity from working context.
    UnpinEntity {
        /// Entity ID to unpin
        entity_id: String,
    },
}

/// Response for session operations.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SessionResponse {
    /// Which operation was performed
    pub operation: String,
    /// Session context (populated for GetContext)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<SessionContextData>,
    /// Pin/unpin result (populated for PinEntity/UnpinEntity)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pin_result: Option<PinResult>,
    /// Helpful hints
    #[serde(default)]
    pub hints: Vec<String>,
}

/// Session context details.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SessionContextData {
    pub verbosity: String,
    pub summary: String,
    #[serde(default)]
    pub last_session_ago: Option<String>,
    pub hot_entities: Vec<HotEntityInfo>,
    pub pending_decisions: Vec<PendingDecisionInfo>,
    #[serde(default)]
    pub world_overview: Option<WorldOverviewInfo>,
}

/// Hot entity in session context.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HotEntityInfo {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    #[serde(default)]
    pub last_accessed: Option<String>,
}

/// Pending decision in session context.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PendingDecisionInfo {
    pub id: String,
    pub description: String,
    pub age: String,
    pub affected_count: usize,
}

/// World overview counts.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WorldOverviewInfo {
    pub character_count: usize,
    pub location_count: usize,
    pub event_count: usize,
    pub scene_count: usize,
    pub relationship_count: usize,
}

/// Pin/unpin operation result.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PinResult {
    pub success: bool,
    pub entity_id: String,
    pub pinned_count: usize,
}
