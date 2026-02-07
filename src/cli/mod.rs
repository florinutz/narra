//! CLI interface for Narra.

pub mod handlers;
pub mod output;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use output::OutputMode;

/// Narra - World state management for fiction writing
#[derive(Parser)]
#[command(name = "narra", version, about, long_about = None)]
pub struct Cli {
    /// Override data directory (default: ~/.narra)
    #[arg(long, env = "NARRA_DATA_PATH", global = true)]
    pub data_path: Option<PathBuf>,

    /// Output as JSON instead of human-readable format
    #[arg(long, global = true)]
    pub json: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start MCP server (stdio transport for Claude Code integration)
    Mcp,

    /// Character management
    #[command(subcommand)]
    Character(CharacterCommands),

    /// Location management
    #[command(subcommand)]
    Location(LocationCommands),

    /// Event management
    #[command(subcommand)]
    Event(EventCommands),

    /// Scene management
    #[command(subcommand)]
    Scene(SceneCommands),

    /// Search entities by text
    Search {
        /// Search query
        query: String,
        /// Use semantic (vector) search
        #[arg(long)]
        semantic: bool,
        /// Use hybrid (keyword + semantic) search
        #[arg(long)]
        hybrid: bool,
        /// Filter by entity type (character, location, event, scene)
        #[arg(long, name = "type")]
        entity_type: Option<String>,
        /// Maximum results
        #[arg(long, default_value = "20")]
        limit: usize,
    },

    /// Knowledge management
    #[command(subcommand)]
    Knowledge(KnowledgeCommands),

    /// Relationship management
    #[command(subcommand)]
    Relationship(RelationshipCommands),

    /// Universe fact management
    #[command(subcommand)]
    Fact(FactCommands),

    /// Note management
    #[command(subcommand)]
    Note(NoteCommands),

    /// Update entity fields
    Update {
        /// Entity ID (e.g., character:alice)
        entity_id: String,
        /// JSON object of fields to update
        #[arg(long, conflicts_with = "set")]
        fields: Option<String>,
        /// Set single field (key=value, repeatable)
        #[arg(long, value_parser = parse_key_val, action = clap::ArgAction::Append)]
        set: Vec<(String, String)>,
    },

    /// Delete entity
    Delete {
        /// Entity ID to delete
        entity_id: String,
        /// Hard delete (bypass protection)
        #[arg(long)]
        hard: bool,
    },

    /// Embedding health report
    Health,

    /// Backfill embeddings for all or specific entity types
    Backfill {
        /// Entity type filter
        #[arg(long, name = "type")]
        entity_type: Option<String>,
    },

    /// Export world data to YAML (NarraImport format)
    Export {
        /// Output file path (defaults to ./narra-export-{date}.yaml)
        #[arg(long, short)]
        output: Option<PathBuf>,
    },

    /// Validate entity consistency
    Validate {
        /// Entity ID (omit for general check)
        entity_id: Option<String>,
    },

    /// Import world data from a YAML file.
    ///
    /// The YAML file contains sections for: characters, locations, events, scenes,
    /// relationships, knowledge, notes, and facts. See the import template at
    /// src/mcp/resources/import_template.yaml for the full format with examples.
    Import {
        /// Path to YAML import file (see import_template.yaml for format)
        file: PathBuf,
        /// Conflict resolution: error (default) reports duplicates, skip silently
        /// ignores them, update merges fields into existing entities
        #[arg(long, default_value = "error")]
        on_conflict: String,
        /// Parse and show entity counts without writing to database
        #[arg(long)]
        dry_run: bool,
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

    /// Narrative analytics and intelligence
    #[command(subcommand)]
    Analyze(AnalyzeCommands),
}

#[derive(Subcommand)]
pub enum AnalyzeCommands {
    /// Network centrality metrics (degree, betweenness)
    Centrality {
        /// Scope filter (e.g. "perceives", "relationships")
        #[arg(long)]
        scope: Option<String>,
        /// Maximum results
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Trace influence propagation from a character
    Influence {
        /// Character ID (e.g. "alice" or "character:alice")
        character: String,
        /// Maximum propagation depth
        #[arg(long, default_value = "3")]
        depth: usize,
    },
    /// Dramatic irony report (knowledge asymmetries)
    Irony {
        /// Focus on specific character
        #[arg(long)]
        character: Option<String>,
        /// Minimum scenes-since threshold
        #[arg(long, default_value = "3")]
        threshold: usize,
    },
    /// Knowledge asymmetries between a specific pair
    Asymmetries {
        /// First character ID
        character_a: String,
        /// Second character ID
        character_b: String,
    },
    /// BelievesWrongly knowledge conflicts
    Conflicts {
        /// Filter by character
        #[arg(long)]
        character: Option<String>,
        /// Maximum results
        #[arg(long, default_value = "50")]
        limit: usize,
    },
    /// Unresolved perception tensions
    Tensions {
        /// Maximum results
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Most-changed entities by arc drift
    ArcDrift {
        /// Filter by entity type (e.g. "character", "knowledge")
        #[arg(long, name = "type")]
        entity_type: Option<String>,
        /// Maximum results
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Narrative situation report (irony, conflicts, tensions, themes)
    SituationReport,
    /// Character dossier (network, knowledge, perceptions)
    Dossier {
        /// Character ID (e.g. "alice" or "character:alice")
        character: String,
    },
    /// Scene planning for a set of characters
    ScenePrep {
        /// Character IDs (comma-separated)
        #[arg(value_delimiter = ',')]
        characters: Vec<String>,
    },
}

#[derive(Subcommand)]
pub enum CharacterCommands {
    /// List all characters
    List,
    /// Get character details
    Get {
        /// Character ID (with or without 'character:' prefix)
        id: String,
    },
    /// Create new character
    Create {
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
}

#[derive(Subcommand)]
pub enum LocationCommands {
    /// List all locations
    List,
    /// Get location details
    Get { id: String },
    /// Create new location
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
    /// List all events (ordered by sequence)
    List,
    /// Get event details
    Get { id: String },
    /// Create new event
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
    /// List all scenes
    List,
    /// Get scene details
    Get { id: String },
    /// Create new scene
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
    /// List knowledge (filtered by character)
    List {
        #[arg(long)]
        character: Option<String>,
    },
    /// Record character knowledge
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
    /// List relationships (filtered by character)
    List {
        #[arg(long)]
        character: Option<String>,
    },
    /// Create relationship between characters
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
    /// List universe facts
    List {
        #[arg(long)]
        category: Option<String>,
        #[arg(long)]
        enforcement: Option<String>,
        #[arg(long)]
        search: Option<String>,
    },
    /// Get fact by ID
    Get { id: String },
    /// Create universe fact
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
    /// Update universe fact
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
    /// Delete universe fact
    Delete { id: String },
    /// Link fact to entity
    Link {
        #[arg(long)]
        fact: String,
        #[arg(long)]
        entity: String,
    },
    /// Unlink fact from entity
    Unlink {
        #[arg(long)]
        fact: String,
        #[arg(long)]
        entity: String,
    },
}

#[derive(Subcommand)]
pub enum NoteCommands {
    /// List notes
    List {
        #[arg(long)]
        entity: Option<String>,
    },
    /// Create note
    Create {
        #[arg(long)]
        title: String,
        #[arg(long)]
        body: String,
        #[arg(long, value_delimiter = ',')]
        attach_to: Vec<String>,
    },
    /// Attach note to entity
    Attach {
        #[arg(long)]
        note: String,
        #[arg(long)]
        entity: String,
    },
    /// Detach note from entity
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
    json: bool,
) -> anyhow::Result<()> {
    let mode = OutputMode::from_json_flag(json);

    match command {
        Commands::Mcp => unreachable!("MCP handled in main"),

        // Character commands
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

        // Location commands
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

        // Event commands
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

        // Scene commands
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

        // Search
        Commands::Search {
            query,
            semantic,
            hybrid,
            entity_type,
            limit,
        } => {
            handlers::search::handle_search(
                ctx,
                query,
                *semantic,
                *hybrid,
                entity_type.as_deref(),
                *limit,
                mode,
            )
            .await?
        }

        // Knowledge commands
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

        // Relationship commands
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

        // Fact commands
        Commands::Fact(cmd) => match cmd {
            FactCommands::List { .. } => handlers::fact::list_facts(ctx, mode).await?,
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

        // Note commands
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

        // Utility commands
        Commands::Update {
            entity_id,
            fields,
            set,
        } => handlers::utility::handle_update(ctx, entity_id, fields.as_deref(), set, mode).await?,
        Commands::Delete { entity_id, hard } => {
            handlers::utility::handle_delete(ctx, entity_id, *hard, mode).await?
        }
        Commands::Health => handlers::utility::handle_health(ctx, mode).await?,
        Commands::Backfill { entity_type } => {
            handlers::utility::handle_backfill(ctx, entity_type.as_deref(), mode).await?
        }
        Commands::Export { output } => {
            handlers::utility::handle_export(ctx, output.as_deref(), mode).await?
        }
        Commands::Validate { entity_id } => {
            handlers::utility::handle_validate(ctx, entity_id.as_deref(), mode).await?
        }
        Commands::Import {
            file,
            on_conflict,
            dry_run,
        } => handlers::import::handle_import(ctx, file, on_conflict, *dry_run, mode).await?,
        Commands::Graph {
            scope,
            depth,
            output,
        } => handlers::utility::handle_graph(ctx, scope, *depth, output.as_deref(), mode).await?,

        // Analyze commands
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
            AnalyzeCommands::SituationReport => {
                handlers::analyze::handle_situation_report(ctx, mode).await?
            }
            AnalyzeCommands::Dossier { character } => {
                handlers::analyze::handle_dossier(ctx, character, mode).await?
            }
            AnalyzeCommands::ScenePrep { characters } => {
                handlers::analyze::handle_scene_prep(ctx, characters, mode).await?
            }
        },
    }

    Ok(())
}
