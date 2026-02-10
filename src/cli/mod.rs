//! CLI interface for Narra.

pub mod handlers;
pub mod output;
pub mod resolve;

use clap::{CommandFactory, Parser, Subcommand};
use std::path::PathBuf;

use output::{DetailLevel, OutputMode};

/// Narra - Narrative intelligence engine for fiction writing
#[derive(Parser)]
#[command(name = "narra", version, about, long_about = None)]
pub struct Cli {
    /// Override data directory (default: ~/.narra)
    #[arg(long, env = "NARRA_DATA_PATH", global = true)]
    pub data_path: Option<PathBuf>,

    /// Output as JSON instead of human-readable format
    #[arg(long, global = true)]
    pub json: bool,

    /// Output as Markdown
    #[arg(long, global = true)]
    pub md: bool,

    /// Brief output (less detail)
    #[arg(long, global = true)]
    pub brief: bool,

    /// Full output (maximum detail)
    #[arg(long, global = true)]
    pub full: bool,

    /// Disable semantic search (use keyword only)
    #[arg(long, global = true)]
    pub no_semantic: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start MCP server (stdio transport for Claude Code integration)
    Mcp,

    /// Deep entity exploration (relationships, knowledge, perceptions, similar)
    Explore {
        /// Entity ID or name
        entity: String,
        /// Skip similar entities section
        #[arg(long)]
        no_similar: bool,
        /// Graph traversal depth for related entities
        #[arg(long, default_value = "2")]
        depth: usize,
    },

    /// Ask a natural language question about the narrative world
    Ask {
        /// The question to ask
        question: String,
        /// Maximum search results
        #[arg(long, default_value = "10")]
        limit: usize,
        /// Include contextual summaries
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        context: bool,
        /// Token budget for context
        #[arg(long, default_value = "2000")]
        budget: usize,
    },

    /// Search across all entities (hybrid by default)
    #[command(alias = "search")]
    Find {
        /// Search query (required for default search, optional with subcommands)
        query: Option<String>,
        /// Use keyword search only (no semantic/vector)
        #[arg(long)]
        keyword_only: bool,
        /// Use semantic (vector) search only
        #[arg(long)]
        semantic_only: bool,
        /// Use cross-encoder re-ranking for better relevance ordering
        #[arg(long)]
        rerank: bool,
        /// Search a specific character facet: identity, psychology, social, or narrative
        #[arg(long)]
        facet: Option<String>,
        /// Filter by entity type (character, location, event, scene, knowledge, note)
        #[arg(long, name = "type")]
        entity_type: Option<String>,
        /// Maximum results
        #[arg(long, default_value = "20")]
        limit: usize,
        #[command(subcommand)]
        subcommand: Option<FindCommands>,
    },

    /// Find shortest connection paths between entities
    Path {
        /// Source entity (ID or name)
        from: String,
        /// Target entity (ID or name)
        to: String,
        /// Maximum hops
        #[arg(long, default_value = "3")]
        max_hops: usize,
        /// Include event connections
        #[arg(long)]
        include_events: bool,
    },

    /// What references this entity
    References {
        /// Entity ID or name
        entity: String,
        /// Filter by referencing entity types (comma-separated)
        #[arg(long, value_delimiter = ',')]
        types: Option<Vec<String>>,
        /// Maximum results
        #[arg(long, default_value = "20")]
        limit: usize,
    },

    /// Mark entity as protected (triggers warnings on impact)
    Protect {
        /// Entity ID or name
        entity: String,
    },

    /// Remove entity protection
    Unprotect {
        /// Entity ID or name
        entity: String,
    },

    /// Get any entity by ID or name
    Get {
        /// Entity ID (type:key) or name for auto-resolution
        entity_id: String,
    },

    /// List entities of a given type
    List {
        /// Entity type (character, location, event, scene, knowledge, relationship, fact, note)
        entity_type: String,
        /// Filter by character (for knowledge, relationship)
        #[arg(long)]
        character: Option<String>,
        /// Filter by category (for facts)
        #[arg(long)]
        category: Option<String>,
        /// Filter by enforcement level (for facts)
        #[arg(long)]
        enforcement: Option<String>,
        /// Filter by attached entity (for notes)
        #[arg(long)]
        entity: Option<String>,
        /// Maximum results
        #[arg(long, default_value = "100")]
        limit: usize,
    },

    /// Create a new entity
    #[command(subcommand)]
    Create(CreateCommands),

    /// Update entity fields (with optional --link / --unlink for facts and notes)
    Update {
        /// Entity ID (e.g., character:alice)
        entity_id: String,
        /// JSON object of fields to update
        #[arg(long, conflicts_with = "set")]
        fields: Option<String>,
        /// Set single field (key=value, repeatable)
        #[arg(long, value_parser = parse_key_val, action = clap::ArgAction::Append)]
        set: Vec<(String, String)>,
        /// Link to entity (for facts: link fact to entity; for notes: attach note to entity)
        #[arg(long)]
        link: Option<String>,
        /// Unlink from entity (for facts: unlink; for notes: detach)
        #[arg(long)]
        unlink: Option<String>,
    },

    /// Delete entity
    Delete {
        /// Entity ID to delete
        entity_id: String,
        /// Hard delete (bypass protection)
        #[arg(long)]
        hard: bool,
    },

    /// Narrative analytics and intelligence
    #[command(subcommand)]
    Analyze(AnalyzeCommands),

    /// World management (status, health, import/export, validation)
    #[command(subcommand)]
    World(WorldCommands),

    /// Session management (context, pin, unpin)
    #[command(subcommand)]
    Session(SessionCommands),

    /// Batch-create entities from YAML (stdin or --file)
    Batch {
        /// Entity type: character, location, event, relationship
        entity_type: String,
        /// Read from file instead of stdin
        #[arg(long)]
        file: Option<String>,
    },

    /// Generate shell completions
    Completions {
        /// Shell type (bash, zsh, fish, elvish, powershell)
        shell: clap_complete::Shell,
    },

    // =========================================================================
    // Legacy subcommands kept for backward compatibility (hidden)
    // =========================================================================
    /// Character management
    #[command(subcommand, hide = true)]
    Character(CharacterCommands),

    /// Location management
    #[command(subcommand, hide = true)]
    Location(LocationCommands),

    /// Event management
    #[command(subcommand, hide = true)]
    Event(EventCommands),

    /// Scene management
    #[command(subcommand, hide = true)]
    Scene(SceneCommands),

    /// Knowledge management
    #[command(subcommand, hide = true)]
    Knowledge(KnowledgeCommands),

    /// Relationship management
    #[command(subcommand, hide = true)]
    Relationship(RelationshipCommands),

    /// Universe fact management
    #[command(subcommand, hide = true)]
    Fact(FactCommands),

    /// Note management
    #[command(subcommand, hide = true)]
    Note(NoteCommands),

    /// Embedding health report
    #[command(hide = true)]
    Health,

    /// Backfill embeddings
    #[command(hide = true)]
    Backfill {
        #[arg(long, name = "type")]
        entity_type: Option<String>,
    },

    /// Export world data to YAML
    #[command(hide = true)]
    Export {
        #[arg(long, short)]
        output: Option<PathBuf>,
    },

    /// Validate entity consistency
    #[command(hide = true)]
    Validate { entity_id: Option<String> },

    /// Import world data from YAML
    #[command(hide = true)]
    Import {
        file: PathBuf,
        #[arg(long, default_value = "error")]
        on_conflict: String,
        #[arg(long)]
        dry_run: bool,
    },

    /// Generate relationship graph
    #[command(hide = true)]
    Graph {
        #[arg(long, default_value = "full")]
        scope: String,
        #[arg(long, default_value = "2")]
        depth: usize,
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
}

// =============================================================================
// New sub-enums
// =============================================================================

#[derive(Subcommand)]
pub enum CreateCommands {
    /// Create a new character
    Character {
        #[arg(long)]
        name: String,
        #[arg(long)]
        role: Option<String>,
        #[arg(long)]
        description: Option<String>,
        #[arg(long, value_delimiter = ',')]
        aliases: Vec<String>,
        /// Profile as JSON object (e.g. '{"wound": ["..."], "secret": ["..."]}')
        #[arg(long)]
        profile: Option<String>,
    },
    /// Create a new location
    Location {
        #[arg(long)]
        name: String,
        #[arg(long)]
        parent: Option<String>,
        #[arg(long)]
        description: Option<String>,
        #[arg(long)]
        loc_type: Option<String>,
    },
    /// Create a new event
    Event {
        #[arg(long)]
        title: String,
        #[arg(long)]
        description: Option<String>,
        #[arg(long)]
        sequence: Option<i32>,
        #[arg(long)]
        date: Option<String>,
    },
    /// Create a new scene
    Scene {
        #[arg(long)]
        title: String,
        #[arg(long)]
        event: String,
        #[arg(long)]
        location: String,
        #[arg(long)]
        summary: Option<String>,
    },
    /// Record character knowledge
    Knowledge {
        #[arg(long)]
        character: String,
        #[arg(long)]
        fact: String,
        #[arg(long, default_value = "knows")]
        certainty: String,
        #[arg(long)]
        method: Option<String>,
        #[arg(long)]
        source: Option<String>,
        #[arg(long)]
        event: Option<String>,
    },
    /// Create a relationship between characters
    Relationship {
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: String,
        #[arg(long, name = "type")]
        rel_type: String,
        #[arg(long)]
        subtype: Option<String>,
        #[arg(long)]
        label: Option<String>,
    },
    /// Record a character's perception of another
    Perception {
        #[arg(long)]
        observer: String,
        #[arg(long)]
        target: String,
        #[arg(long)]
        perception: String,
        #[arg(long)]
        feelings: Option<String>,
        #[arg(long)]
        tension: Option<i32>,
        #[arg(long, value_delimiter = ',')]
        rel_types: Vec<String>,
        #[arg(long)]
        subtype: Option<String>,
        #[arg(long)]
        history: Option<String>,
    },
    /// Create a universe fact
    Fact {
        #[arg(long)]
        title: String,
        #[arg(long)]
        description: String,
        #[arg(long, value_delimiter = ',')]
        categories: Vec<String>,
        #[arg(long)]
        enforcement: Option<String>,
    },
    /// Create a note
    Note {
        #[arg(long)]
        title: String,
        #[arg(long)]
        body: String,
        #[arg(long, value_delimiter = ',')]
        attach_to: Vec<String>,
    },
}

#[derive(Subcommand)]
pub enum FindCommands {
    /// Cross-type semantic search by meaning
    Join {
        /// Search query
        query: String,
        /// Filter by entity type
        #[arg(long, name = "type")]
        entity_type: Option<String>,
        /// Maximum results
        #[arg(long, default_value = "10")]
        limit: usize,
    },
    /// Semantic search within knowledge/facts
    Knowledge {
        /// Search query
        query: String,
        /// Filter by character (name or ID)
        #[arg(long)]
        character: Option<String>,
        /// Maximum results
        #[arg(long, default_value = "10")]
        limit: usize,
    },
    /// Graph proximity + semantic similarity search
    Graph {
        /// Entity to center search on (name or ID)
        entity: String,
        /// Search query
        query: String,
        /// Graph traversal depth
        #[arg(long, default_value = "2")]
        hops: usize,
        /// Filter by entity type
        #[arg(long, name = "type")]
        entity_type: Option<String>,
        /// Maximum results
        #[arg(long, default_value = "10")]
        limit: usize,
    },
    /// Search perceptions/perspectives by meaning
    Perspectives {
        /// Search query
        query: String,
        /// Filter by observer (name or ID)
        #[arg(long)]
        observer: Option<String>,
        /// Filter by target (name or ID)
        #[arg(long)]
        target: Option<String>,
        /// Maximum results
        #[arg(long, default_value = "10")]
        limit: usize,
    },
    /// Find relationships similar to a reference pair
    #[command(alias = "similar")]
    SimilarDynamics {
        /// Source observer (name or ID)
        observer: String,
        /// Source target (name or ID)
        target: String,
        /// Bias search toward this text
        #[arg(long)]
        bias: Option<String>,
        /// Edge type filter (perceives, relates_to)
        #[arg(long)]
        edge_type: Option<String>,
        /// Maximum results
        #[arg(long, default_value = "10")]
        limit: usize,
    },
}

#[derive(Subcommand)]
pub enum WorldCommands {
    /// World overview dashboard (entity counts, embedding coverage)
    Status,
    /// Embedding health report
    Health,
    /// Backfill embeddings for all or specific entity types
    Backfill {
        /// Entity type filter
        #[arg(long, name = "type")]
        entity_type: Option<String>,
        /// Force re-embedding all entities (use after switching embedding model)
        #[arg(long)]
        force: bool,
    },
    /// Export world data to YAML
    Export {
        /// Output file path (defaults to ./narra-export-{date}.yaml)
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
    /// Import world data from a YAML file
    Import {
        /// Path to YAML import file
        file: PathBuf,
        /// Conflict resolution: error, skip, or update
        #[arg(long, default_value = "error")]
        on_conflict: String,
        /// Parse and show entity counts without writing to database
        #[arg(long)]
        dry_run: bool,
    },
    /// Validate entity consistency
    Validate {
        /// Entity ID (omit for general check)
        entity_id: Option<String>,
    },
    /// Generate relationship graph (Mermaid format)
    Graph {
        /// Scope: 'full' or 'character:ID'
        #[arg(long, default_value = "full")]
        scope: String,
        /// Traversal depth
        #[arg(long, default_value = "2")]
        depth: usize,
        /// Output file path
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
    /// Create baseline arc snapshots for entities with embeddings
    BaselineArcs {
        /// Entity type filter (character or knowledge; omit for both)
        #[arg(long, name = "type")]
        entity_type: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum SessionCommands {
    /// Show session context (hot entities, pending decisions, world overview)
    Context,
    /// Pin an entity to persistent session context
    Pin {
        /// Entity ID or name
        entity: String,
    },
    /// Unpin an entity from session context
    Unpin {
        /// Entity ID or name
        entity: String,
    },
}

// =============================================================================
// Analyze commands (unchanged from original)
// =============================================================================

#[derive(Subcommand)]
pub enum AnalyzeCommands {
    /// Network centrality metrics (degree, betweenness)
    Centrality {
        #[arg(long)]
        scope: Option<String>,
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Trace influence propagation from a character
    Influence {
        character: String,
        #[arg(long, default_value = "3")]
        depth: usize,
    },
    /// Dramatic irony report (knowledge asymmetries)
    Irony {
        #[arg(long)]
        character: Option<String>,
        #[arg(long, default_value = "3")]
        threshold: usize,
    },
    /// Knowledge asymmetries between a specific pair
    Asymmetries {
        character_a: String,
        character_b: String,
    },
    /// BelievesWrongly knowledge conflicts
    Conflicts {
        #[arg(long)]
        character: Option<String>,
        #[arg(long, default_value = "50")]
        limit: usize,
    },
    /// Unresolved perception tensions
    Tensions {
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Most-changed entities by arc drift
    ArcDrift {
        #[arg(long, name = "type")]
        entity_type: Option<String>,
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Thematic clustering (k-means on entity embeddings)
    Themes {
        /// Entity types to cluster (comma-separated: char,event,location,scene,knowledge)
        #[arg(long, value_delimiter = ',')]
        types: Option<Vec<String>>,
        /// Number of clusters (auto if not specified)
        #[arg(long)]
        clusters: Option<usize>,
    },
    /// Missing entity types in thematic clusters
    ThematicGaps {
        /// Minimum cluster size to analyze
        #[arg(long, default_value = "3")]
        min_size: usize,
        /// Expected types per cluster (comma-separated)
        #[arg(long, value_delimiter = ',')]
        expected_types: Option<Vec<String>>,
    },
    /// Temporal knowledge: what a character knows at a point in time
    Temporal {
        /// Character (ID or name)
        character: String,
        /// Anchor to a specific event (ID or name)
        #[arg(long)]
        event: Option<String>,
    },
    /// Investigate contradictions across connected entities
    Contradictions {
        /// Entity (ID or name)
        entity: String,
        /// Graph traversal depth
        #[arg(long, default_value = "3")]
        depth: usize,
    },
    /// Perception gap: how wrong is observer about target
    PerceptionGap {
        /// Observer character (ID or name)
        observer: String,
        /// Target character (ID or name)
        target: String,
    },
    /// Perception matrix: all observers' accuracy for a target
    PerceptionMatrix {
        /// Target character (ID or name)
        target: String,
        /// Maximum observers
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Perception shift: observer's view evolution over time
    PerceptionShift {
        /// Observer character (ID or name)
        observer: String,
        /// Target character (ID or name)
        target: String,
    },
    /// Arc history: entity embedding evolution timeline
    ArcHistory {
        /// Entity (ID or name)
        entity: String,
        /// Maximum snapshots
        #[arg(long, default_value = "50")]
        limit: usize,
    },
    /// Arc comparison: compare two entity trajectories
    ArcCompare {
        /// First entity (ID or name)
        entity_a: String,
        /// Second entity (ID or name)
        entity_b: String,
        /// Time window (e.g., "recent:10")
        #[arg(long)]
        window: Option<String>,
    },
    /// Arc moment: entity state at a point in time
    ArcMoment {
        /// Entity (ID or name)
        entity: String,
        /// Event ID to anchor the moment
        #[arg(long)]
        event: Option<String>,
    },
    /// What-if: preview impact of a character learning a fact
    WhatIf {
        /// Character (ID or name)
        character: String,
        /// Knowledge fact ID
        fact: String,
        /// Certainty level (knows, suspects, heard_rumor)
        #[arg(long)]
        certainty: Option<String>,
    },
    /// Impact cascade analysis for entity changes
    Impact {
        /// Entity (ID or name)
        entity: String,
        /// Description of the change
        #[arg(long)]
        description: Option<String>,
    },
    /// Narrative situation report (irony, conflicts, tensions, themes)
    SituationReport,
    /// Character dossier (network, knowledge, perceptions)
    Dossier { character: String },
    /// Scene planning for a set of characters
    ScenePrep {
        #[arg(value_delimiter = ',')]
        characters: Vec<String>,
    },
    /// Growth vector: where is an entity heading based on arc snapshots
    GrowthVector {
        /// Entity (ID or name)
        entity: String,
        /// Maximum trajectory neighbors
        #[arg(long, default_value = "5")]
        limit: usize,
    },
    /// Misperception vector: what does observer get wrong about target
    Misperception {
        /// Observer character (ID or name)
        observer: String,
        /// Target character (ID or name)
        target: String,
        /// Maximum neighbors
        #[arg(long, default_value = "5")]
        limit: usize,
    },
    /// Convergence analysis: are two entities becoming more alike over time
    Convergence {
        /// First entity (ID or name)
        entity_a: String,
        /// Second entity (ID or name)
        entity_b: String,
        /// Maximum snapshots to analyze
        #[arg(long)]
        window: Option<usize>,
    },
    /// Semantic midpoint: find entities bridging two concepts
    Midpoint {
        /// First entity (ID or name)
        entity_a: String,
        /// Second entity (ID or name)
        entity_b: String,
        /// Maximum results
        #[arg(long, default_value = "5")]
        limit: usize,
    },
    /// Character facet diagnostics: view all facet statuses and inter-facet similarities
    Facets {
        /// Character ID or name
        character: String,
    },
}

// =============================================================================
// Legacy sub-enums (hidden, backward compat)
// =============================================================================

#[derive(Subcommand)]
pub enum CharacterCommands {
    List,
    Get {
        id: String,
    },
    Create {
        #[arg(long)]
        name: String,
        #[arg(long)]
        role: Option<String>,
        #[arg(long)]
        description: Option<String>,
        #[arg(long, value_delimiter = ',')]
        aliases: Vec<String>,
        #[arg(long)]
        profile: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum LocationCommands {
    List,
    Get {
        id: String,
    },
    Create {
        #[arg(long)]
        name: String,
        #[arg(long)]
        parent: Option<String>,
        #[arg(long)]
        description: Option<String>,
        #[arg(long)]
        loc_type: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum EventCommands {
    List,
    Get {
        id: String,
    },
    Create {
        #[arg(long)]
        title: String,
        #[arg(long)]
        description: Option<String>,
        #[arg(long)]
        sequence: Option<i32>,
        #[arg(long)]
        date: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum SceneCommands {
    List,
    Get {
        id: String,
    },
    Create {
        #[arg(long)]
        title: String,
        #[arg(long)]
        event: String,
        #[arg(long)]
        location: String,
        #[arg(long)]
        summary: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum KnowledgeCommands {
    List {
        #[arg(long)]
        character: Option<String>,
    },
    Record {
        #[arg(long)]
        character: String,
        #[arg(long)]
        fact: String,
        #[arg(long, default_value = "knows")]
        certainty: String,
        #[arg(long)]
        method: Option<String>,
        #[arg(long)]
        source: Option<String>,
        #[arg(long)]
        event: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum RelationshipCommands {
    List {
        #[arg(long)]
        character: Option<String>,
    },
    Create {
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: String,
        #[arg(long, name = "type")]
        rel_type: String,
        #[arg(long)]
        subtype: Option<String>,
        #[arg(long)]
        label: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum FactCommands {
    List {
        #[arg(long)]
        category: Option<String>,
        #[arg(long)]
        enforcement: Option<String>,
        #[arg(long)]
        search: Option<String>,
    },
    Get {
        id: String,
    },
    Create {
        #[arg(long)]
        title: String,
        #[arg(long)]
        description: String,
        #[arg(long, value_delimiter = ',')]
        categories: Vec<String>,
        #[arg(long)]
        enforcement: Option<String>,
    },
    Update {
        id: String,
        #[arg(long)]
        title: Option<String>,
        #[arg(long)]
        description: Option<String>,
        #[arg(long, value_delimiter = ',')]
        categories: Vec<String>,
        #[arg(long)]
        enforcement: Option<String>,
    },
    Delete {
        id: String,
    },
    Link {
        #[arg(long)]
        fact: String,
        #[arg(long)]
        entity: String,
    },
    Unlink {
        #[arg(long)]
        fact: String,
        #[arg(long)]
        entity: String,
    },
}

#[derive(Subcommand)]
pub enum NoteCommands {
    List {
        #[arg(long)]
        entity: Option<String>,
    },
    Create {
        #[arg(long)]
        title: String,
        #[arg(long)]
        body: String,
        #[arg(long, value_delimiter = ',')]
        attach_to: Vec<String>,
    },
    Attach {
        #[arg(long)]
        note: String,
        #[arg(long)]
        entity: String,
    },
    Detach {
        #[arg(long)]
        note: String,
        #[arg(long)]
        entity: String,
    },
}

/// Parse key=value pairs for --set flag
fn parse_key_val(s: &str) -> Result<(String, String), String> {
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid key=value: no '=' found in '{}'", s))?;
    Ok((s[..pos].to_string(), s[pos + 1..].to_string()))
}

/// Execute a CLI command, dispatching to the appropriate handler.
pub async fn execute(
    command: &Commands,
    ctx: &crate::init::AppContext,
    mode: OutputMode,
    detail: DetailLevel,
    no_semantic: bool,
) -> anyhow::Result<()> {
    let _ = detail; // Used in future sessions for controlling output verbosity

    match command {
        Commands::Mcp => unreachable!("MCP handled in main"),

        // =====================================================================
        // New intent-based commands
        // =====================================================================
        Commands::Path {
            from,
            to,
            max_hops,
            include_events,
        } => {
            handlers::path::handle_path(
                ctx,
                from,
                to,
                *max_hops,
                *include_events,
                mode,
                no_semantic,
            )
            .await?
        }

        Commands::References {
            entity,
            types,
            limit,
        } => {
            handlers::path::handle_references(ctx, entity, types.clone(), *limit, mode, no_semantic)
                .await?
        }

        Commands::Protect { entity } => {
            handlers::entity::handle_protect(ctx, entity, mode, no_semantic).await?
        }

        Commands::Unprotect { entity } => {
            handlers::entity::handle_unprotect(ctx, entity, mode, no_semantic).await?
        }

        Commands::Find {
            query,
            keyword_only,
            semantic_only,
            rerank,
            facet,
            entity_type,
            limit,
            subcommand,
        } => match subcommand {
            Some(FindCommands::Join {
                query: q,
                entity_type: et,
                limit: l,
            }) => handlers::find::handle_semantic_join(ctx, q, et.as_deref(), *l, mode).await?,
            Some(FindCommands::Knowledge {
                query: q,
                character,
                limit: l,
            }) => {
                handlers::find::handle_semantic_knowledge(
                    ctx,
                    q,
                    character.as_deref(),
                    *l,
                    mode,
                    no_semantic,
                )
                .await?
            }
            Some(FindCommands::Graph {
                entity,
                query: q,
                hops,
                entity_type: et,
                limit: l,
            }) => {
                handlers::find::handle_semantic_graph_search(
                    ctx,
                    entity,
                    q,
                    *hops,
                    et.as_deref(),
                    *l,
                    mode,
                    no_semantic,
                )
                .await?
            }
            Some(FindCommands::Perspectives {
                query: q,
                observer,
                target,
                limit: l,
            }) => {
                handlers::perception::handle_perspective_search(
                    ctx,
                    q,
                    observer.as_deref(),
                    target.as_deref(),
                    *l,
                    mode,
                    no_semantic,
                )
                .await?
            }
            Some(FindCommands::SimilarDynamics {
                observer,
                target,
                bias,
                edge_type,
                limit: l,
            }) => {
                handlers::relationship::handle_similar_relationships(
                    ctx,
                    observer,
                    target,
                    edge_type.as_deref(),
                    bias.as_deref(),
                    *l,
                    mode,
                    no_semantic,
                )
                .await?
            }
            None => {
                let q = query.as_deref().unwrap_or("");
                if q.is_empty() {
                    anyhow::bail!("Search query is required. Usage: narra find \"query\" or narra find <subcommand>");
                }
                handlers::find::handle_find(
                    ctx,
                    q,
                    *keyword_only,
                    *semantic_only,
                    *rerank,
                    facet.as_deref(),
                    no_semantic,
                    entity_type.as_deref(),
                    *limit,
                    mode,
                )
                .await?
            }
        },

        Commands::Explore {
            entity,
            no_similar,
            depth,
        } => {
            handlers::explore::handle_explore(ctx, entity, *no_similar, *depth, mode, no_semantic)
                .await?
        }

        Commands::Ask {
            question,
            limit,
            context,
            budget,
        } => {
            handlers::ask::handle_ask(ctx, question, *limit, *context, *budget, mode, no_semantic)
                .await?
        }

        Commands::Get { entity_id } => {
            handlers::entity::handle_get(ctx, entity_id, mode, no_semantic).await?
        }

        Commands::List {
            entity_type,
            character,
            category,
            enforcement,
            entity,
            limit,
        } => {
            handlers::entity::handle_list(
                ctx,
                entity_type,
                character.as_deref(),
                category.as_deref(),
                enforcement.as_deref(),
                entity.as_deref(),
                *limit,
                mode,
            )
            .await?
        }

        Commands::Create(cmd) => handle_create(cmd, ctx, mode).await?,

        Commands::Update {
            entity_id,
            fields,
            set,
            link,
            unlink,
        } => {
            handlers::utility::handle_update(
                ctx,
                entity_id,
                fields.as_deref(),
                set,
                link.as_deref(),
                unlink.as_deref(),
                mode,
            )
            .await?
        }

        Commands::Delete { entity_id, hard } => {
            handlers::utility::handle_delete(ctx, entity_id, *hard, mode).await?
        }

        // =====================================================================
        // World commands
        // =====================================================================
        Commands::World(cmd) => match cmd {
            WorldCommands::Status => handlers::world::handle_status(ctx, mode).await?,
            WorldCommands::Health => handlers::world::handle_health(ctx, mode).await?,
            WorldCommands::Backfill { entity_type, force } => {
                handlers::world::handle_backfill(ctx, entity_type.as_deref(), *force, mode).await?
            }
            WorldCommands::Export { output } => {
                handlers::world::handle_export(ctx, output.as_deref(), mode).await?
            }
            WorldCommands::Import {
                file,
                on_conflict,
                dry_run,
            } => handlers::world::handle_import(ctx, file, on_conflict, *dry_run, mode).await?,
            WorldCommands::Validate { entity_id } => {
                handlers::world::handle_validate(ctx, entity_id.as_deref(), mode).await?
            }
            WorldCommands::Graph {
                scope,
                depth,
                output,
            } => handlers::world::handle_graph(ctx, scope, *depth, output.as_deref(), mode).await?,
            WorldCommands::BaselineArcs { entity_type } => {
                handlers::world::handle_baseline_arcs(ctx, entity_type.as_deref(), mode).await?
            }
        },

        // =====================================================================
        // Session commands
        // =====================================================================
        Commands::Session(cmd) => match cmd {
            SessionCommands::Context => handlers::session::handle_context(ctx, mode).await?,
            SessionCommands::Pin { entity } => {
                handlers::session::handle_pin(ctx, entity, mode, no_semantic).await?
            }
            SessionCommands::Unpin { entity } => {
                handlers::session::handle_unpin(ctx, entity, mode, no_semantic).await?
            }
        },

        // =====================================================================
        // Batch create
        // =====================================================================
        Commands::Batch { entity_type, file } => {
            handlers::batch::handle_batch_create(ctx, entity_type, file.as_deref(), mode).await?
        }

        // =====================================================================
        // Shell completions (no AppContext needed, but we have it here)
        // =====================================================================
        Commands::Completions { shell } => {
            clap_complete::generate(*shell, &mut Cli::command(), "narra", &mut std::io::stdout());
        }

        // =====================================================================
        // Analyze commands (unchanged)
        // =====================================================================
        Commands::Analyze(cmd) => match cmd {
            AnalyzeCommands::Centrality { scope, limit } => {
                handlers::analyze::handle_centrality(ctx, scope.as_deref(), *limit, mode).await?
            }
            AnalyzeCommands::Influence { character, depth } => {
                handlers::analyze::handle_influence(ctx, character, *depth, mode).await?
            }
            AnalyzeCommands::Irony {
                character,
                threshold,
            } => {
                handlers::analyze::handle_irony(ctx, character.as_deref(), *threshold, mode).await?
            }
            AnalyzeCommands::Asymmetries {
                character_a,
                character_b,
            } => handlers::analyze::handle_asymmetries(ctx, character_a, character_b, mode).await?,
            AnalyzeCommands::Conflicts { character, limit } => {
                handlers::analyze::handle_conflicts(ctx, character.as_deref(), *limit, mode).await?
            }
            AnalyzeCommands::Tensions { limit } => {
                handlers::analyze::handle_tensions(ctx, *limit, mode).await?
            }
            AnalyzeCommands::ArcDrift { entity_type, limit } => {
                handlers::analyze::handle_arc_drift(ctx, entity_type.as_deref(), *limit, mode)
                    .await?
            }
            AnalyzeCommands::Themes { types, clusters } => {
                handlers::analyze::handle_themes(ctx, types.clone(), *clusters, mode).await?
            }
            AnalyzeCommands::ThematicGaps {
                min_size,
                expected_types,
            } => {
                handlers::analyze::handle_thematic_gaps(
                    ctx,
                    *min_size,
                    expected_types.clone(),
                    mode,
                )
                .await?
            }
            AnalyzeCommands::Temporal { character, event } => {
                handlers::analyze::handle_temporal(ctx, character, event.clone(), mode, no_semantic)
                    .await?
            }
            AnalyzeCommands::Contradictions { entity, depth } => {
                handlers::analyze::handle_contradictions(ctx, entity, *depth, mode, no_semantic)
                    .await?
            }
            AnalyzeCommands::PerceptionGap { observer, target } => {
                handlers::perception::handle_perception_gap(
                    ctx,
                    observer,
                    target,
                    mode,
                    no_semantic,
                )
                .await?
            }
            AnalyzeCommands::PerceptionMatrix { target, limit } => {
                handlers::perception::handle_perception_matrix(
                    ctx,
                    target,
                    *limit,
                    mode,
                    no_semantic,
                )
                .await?
            }
            AnalyzeCommands::PerceptionShift { observer, target } => {
                handlers::perception::handle_perception_shift(
                    ctx,
                    observer,
                    target,
                    mode,
                    no_semantic,
                )
                .await?
            }
            AnalyzeCommands::ArcHistory { entity, limit } => {
                handlers::arc::handle_arc_history(ctx, entity, *limit, mode, no_semantic).await?
            }
            AnalyzeCommands::ArcCompare {
                entity_a,
                entity_b,
                window,
            } => {
                handlers::arc::handle_arc_compare(
                    ctx,
                    entity_a,
                    entity_b,
                    window.clone(),
                    mode,
                    no_semantic,
                )
                .await?
            }
            AnalyzeCommands::ArcMoment { entity, event } => {
                handlers::arc::handle_arc_moment(ctx, entity, event.clone(), mode, no_semantic)
                    .await?
            }
            AnalyzeCommands::WhatIf {
                character,
                fact,
                certainty,
            } => {
                handlers::arc::handle_what_if(
                    ctx,
                    character,
                    fact,
                    certainty.clone(),
                    mode,
                    no_semantic,
                )
                .await?
            }
            AnalyzeCommands::Impact {
                entity,
                description,
            } => {
                handlers::analyze::handle_impact(
                    ctx,
                    entity,
                    description.clone(),
                    mode,
                    no_semantic,
                )
                .await?
            }
            AnalyzeCommands::SituationReport => {
                handlers::analyze::handle_situation_report(ctx, mode).await?
            }
            AnalyzeCommands::Dossier { character } => {
                handlers::analyze::handle_dossier(ctx, character, mode).await?
            }
            AnalyzeCommands::ScenePrep { characters } => {
                handlers::analyze::handle_scene_prep(ctx, characters, mode).await?
            }
            AnalyzeCommands::GrowthVector { entity, limit } => {
                handlers::analyze::handle_growth_vector(ctx, entity, *limit, mode, no_semantic)
                    .await?
            }
            AnalyzeCommands::Misperception {
                observer,
                target,
                limit,
            } => {
                handlers::analyze::handle_misperception(
                    ctx,
                    observer,
                    target,
                    *limit,
                    mode,
                    no_semantic,
                )
                .await?
            }
            AnalyzeCommands::Convergence {
                entity_a,
                entity_b,
                window,
            } => {
                handlers::analyze::handle_convergence(
                    ctx,
                    entity_a,
                    entity_b,
                    *window,
                    mode,
                    no_semantic,
                )
                .await?
            }
            AnalyzeCommands::Midpoint {
                entity_a,
                entity_b,
                limit,
            } => {
                handlers::analyze::handle_midpoint(
                    ctx,
                    entity_a,
                    entity_b,
                    *limit,
                    mode,
                    no_semantic,
                )
                .await?
            }
            AnalyzeCommands::Facets { character } => {
                handlers::analyze::handle_facets(ctx, character, mode, no_semantic).await?
            }
        },

        // =====================================================================
        // Legacy commands (hidden, backward compat)
        // =====================================================================
        Commands::Character(cmd) => match cmd {
            CharacterCommands::List => handlers::entity::list_characters(ctx, mode).await?,
            CharacterCommands::Get { id } => handlers::entity::get_character(ctx, id, mode).await?,
            CharacterCommands::Create {
                name,
                role,
                description,
                aliases,
                profile,
            } => {
                handlers::entity::create_character(
                    ctx,
                    name,
                    role.as_deref(),
                    description.as_deref(),
                    aliases,
                    profile.as_deref(),
                    mode,
                )
                .await?
            }
        },

        Commands::Location(cmd) => match cmd {
            LocationCommands::List => handlers::entity::list_locations(ctx, mode).await?,
            LocationCommands::Get { id } => handlers::entity::get_location(ctx, id, mode).await?,
            LocationCommands::Create {
                name,
                parent,
                description,
                loc_type,
            } => {
                handlers::entity::create_location(
                    ctx,
                    name,
                    description.as_deref(),
                    parent.as_deref(),
                    loc_type.as_deref(),
                    mode,
                )
                .await?
            }
        },

        Commands::Event(cmd) => match cmd {
            EventCommands::List => handlers::entity::list_events(ctx, mode).await?,
            EventCommands::Get { id } => handlers::entity::get_event(ctx, id, mode).await?,
            EventCommands::Create {
                title,
                description,
                sequence,
                date,
            } => {
                handlers::entity::create_event(
                    ctx,
                    title,
                    description.as_deref(),
                    *sequence,
                    date.as_deref(),
                    mode,
                )
                .await?
            }
        },

        Commands::Scene(cmd) => match cmd {
            SceneCommands::List => handlers::entity::list_scenes(ctx, mode).await?,
            SceneCommands::Get { id } => handlers::entity::get_scene(ctx, id, mode).await?,
            SceneCommands::Create {
                title,
                event,
                location,
                summary,
            } => {
                handlers::entity::create_scene(
                    ctx,
                    title,
                    event,
                    location,
                    summary.as_deref(),
                    mode,
                )
                .await?
            }
        },

        Commands::Knowledge(cmd) => match cmd {
            KnowledgeCommands::List { character } => {
                handlers::knowledge::list_knowledge(ctx, character.as_deref(), mode).await?
            }
            KnowledgeCommands::Record {
                character,
                fact,
                certainty,
                method,
                source,
                event,
            } => {
                handlers::knowledge::record_knowledge(
                    ctx,
                    character,
                    fact,
                    certainty,
                    method.as_deref(),
                    source.as_deref(),
                    event.as_deref(),
                    mode,
                )
                .await?
            }
        },

        Commands::Relationship(cmd) => match cmd {
            RelationshipCommands::List { character } => {
                handlers::relationship::list_relationships(ctx, character.as_deref(), mode).await?
            }
            RelationshipCommands::Create {
                from,
                to,
                rel_type,
                subtype,
                label,
            } => {
                handlers::relationship::create_relationship(
                    ctx,
                    from,
                    to,
                    rel_type,
                    subtype.as_deref(),
                    label.as_deref(),
                    mode,
                )
                .await?
            }
        },

        Commands::Fact(cmd) => match cmd {
            FactCommands::List {
                category,
                enforcement,
                ..
            } => {
                handlers::fact::list_facts(
                    ctx,
                    None,
                    category.as_deref(),
                    enforcement.as_deref(),
                    mode,
                )
                .await?
            }
            FactCommands::Get { id } => handlers::fact::get_fact(ctx, id, mode).await?,
            FactCommands::Create {
                title,
                description,
                categories,
                enforcement,
            } => {
                handlers::fact::create_fact(
                    ctx,
                    title,
                    description,
                    categories,
                    enforcement.as_deref(),
                    mode,
                )
                .await?
            }
            FactCommands::Update {
                id,
                title,
                description,
                categories,
                enforcement,
            } => {
                handlers::fact::update_fact(
                    ctx,
                    id,
                    title.as_deref(),
                    description.as_deref(),
                    categories,
                    enforcement.as_deref(),
                    mode,
                )
                .await?
            }
            FactCommands::Delete { id } => handlers::fact::delete_fact(ctx, id, mode).await?,
            FactCommands::Link { fact, entity } => {
                handlers::fact::link_fact(ctx, fact, entity, mode).await?
            }
            FactCommands::Unlink { fact, entity } => {
                handlers::fact::unlink_fact(ctx, fact, entity, mode).await?
            }
        },

        Commands::Note(cmd) => match cmd {
            NoteCommands::List { entity } => {
                handlers::note::list_notes(ctx, entity.as_deref(), mode).await?
            }
            NoteCommands::Create {
                title,
                body,
                attach_to,
            } => handlers::note::create_note(ctx, title, body, attach_to, mode).await?,
            NoteCommands::Attach { note, entity } => {
                handlers::note::attach_note(ctx, note, entity, mode).await?
            }
            NoteCommands::Detach { note, entity } => {
                handlers::note::detach_note(ctx, note, entity, mode).await?
            }
        },

        // Legacy top-level aliases → world commands
        Commands::Health => handlers::world::handle_health(ctx, mode).await?,
        Commands::Backfill { entity_type } => {
            handlers::world::handle_backfill(ctx, entity_type.as_deref(), false, mode).await?
        }
        Commands::Export { output } => {
            handlers::world::handle_export(ctx, output.as_deref(), mode).await?
        }
        Commands::Validate { entity_id } => {
            handlers::world::handle_validate(ctx, entity_id.as_deref(), mode).await?
        }
        Commands::Import {
            file,
            on_conflict,
            dry_run,
        } => handlers::world::handle_import(ctx, file, on_conflict, *dry_run, mode).await?,
        Commands::Graph {
            scope,
            depth,
            output,
        } => handlers::world::handle_graph(ctx, scope, *depth, output.as_deref(), mode).await?,
    }

    Ok(())
}

/// Dispatch create subcommands to their handlers.
async fn handle_create(
    cmd: &CreateCommands,
    ctx: &crate::init::AppContext,
    mode: OutputMode,
) -> anyhow::Result<()> {
    match cmd {
        CreateCommands::Character {
            name,
            role,
            description,
            aliases,
            profile,
        } => {
            handlers::entity::create_character(
                ctx,
                name,
                role.as_deref(),
                description.as_deref(),
                aliases,
                profile.as_deref(),
                mode,
            )
            .await
        }
        CreateCommands::Location {
            name,
            parent,
            description,
            loc_type,
        } => {
            handlers::entity::create_location(
                ctx,
                name,
                description.as_deref(),
                parent.as_deref(),
                loc_type.as_deref(),
                mode,
            )
            .await
        }
        CreateCommands::Event {
            title,
            description,
            sequence,
            date,
        } => {
            handlers::entity::create_event(
                ctx,
                title,
                description.as_deref(),
                *sequence,
                date.as_deref(),
                mode,
            )
            .await
        }
        CreateCommands::Scene {
            title,
            event,
            location,
            summary,
        } => {
            handlers::entity::create_scene(ctx, title, event, location, summary.as_deref(), mode)
                .await
        }
        CreateCommands::Knowledge {
            character,
            fact,
            certainty,
            method,
            source,
            event,
        } => {
            handlers::knowledge::record_knowledge(
                ctx,
                character,
                fact,
                certainty,
                method.as_deref(),
                source.as_deref(),
                event.as_deref(),
                mode,
            )
            .await
        }
        CreateCommands::Relationship {
            from,
            to,
            rel_type,
            subtype,
            label,
        } => {
            handlers::relationship::create_relationship(
                ctx,
                from,
                to,
                rel_type,
                subtype.as_deref(),
                label.as_deref(),
                mode,
            )
            .await
        }
        CreateCommands::Perception {
            observer,
            target,
            perception,
            feelings,
            tension,
            rel_types,
            subtype,
            history,
        } => {
            handlers::perception::create_perception(
                ctx,
                observer,
                target,
                perception,
                feelings.as_deref(),
                *tension,
                rel_types,
                subtype.as_deref(),
                history.as_deref(),
                mode,
            )
            .await
        }
        CreateCommands::Fact {
            title,
            description,
            categories,
            enforcement,
        } => {
            handlers::fact::create_fact(
                ctx,
                title,
                description,
                categories,
                enforcement.as_deref(),
                mode,
            )
            .await
        }
        CreateCommands::Note {
            title,
            body,
            attach_to,
        } => handlers::note::create_note(ctx, title, body, attach_to, mode).await,
    }
}
