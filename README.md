# Narra

Narrative intelligence engine for AI-assisted fiction writing.

[![CI](https://github.com/florinutz/narra/actions/workflows/ci.yml/badge.svg)](https://github.com/florinutz/narra/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/florinutz/narra)](https://github.com/florinutz/narra/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Narra is a structured database for fictional universes — characters, locations, events, scenes, relationships, knowledge, perceptions, facts, and notes — all queryable by meaning via local embeddings.

It goes beyond CRUD. Narra detects dramatic irony, tracks character evolution through embedding drift, measures perception gaps between characters, surfaces unresolved tensions, clusters emergent themes, finds knowledge conflicts, traces influence propagation through relationship networks, and previews what-if scenarios — all without external API calls.

Everything runs locally: embedded SurrealDB with RocksDB persistence, BGE-small-en-v1.5 embeddings via fastembed, single binary, zero runtime dependencies. Use it as a **CLI tool** or as an **MCP server** for AI-assisted writing workflows.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [CLI Command Reference](#cli-command-reference)
- [CLI Walkthrough](#cli-walkthrough)
- [MCP Server](#mcp-server)
- [Data Storage](#data-storage)
- [Architecture](#architecture)
- [Development](#development)
- [License](#license)

## Features

### World Building

- **Characters** — flexible profile system with structured keys (wound, desire, contradiction, secret) plus arbitrary custom keys
- **Locations** — hierarchical parent-child structure with types
- **Events** — sequence-ordered timeline with optional dates and date precision
- **Scenes** — anchored to event + location, with typed participants
- **Relationships** — 7 built-in types (family, romantic, professional, social, antagonistic, mentorship, custom) with subtypes and labels
- **Knowledge** — append-only ledger with 7 certainty levels (knows, suspects, believes_wrongly, uncertain, assumes, denies, forgotten) and 8 learning methods (told, overheard, witnessed, discovered, deduced, read, remembered, initial), full provenance (source character, event)
- **Perceptions** — asymmetric: A's view of B is independent of B's view of A, with feelings, tension level, and history
- **Universe facts** — world rules with enforcement levels: informational (context only), warning (flags violations), strict (blocks mutations)
- **Notes** — freeform text attachable to any entity
- **Import/Export** — round-trip YAML with dependency-ordered processing (characters → locations → events → scenes → relationships → knowledge → notes → facts)

### Narrative Intelligence

| Capability | What it does |
|---|---|
| **Semantic search** | Find entities by meaning, not keywords ("characters struggling with duty") |
| **Hybrid search** | Combine keyword matching with semantic similarity ("betrayal scenes with Bob") |
| **Dramatic irony** | Detect knowledge asymmetries — what some characters know that others don't |
| **Perception gaps** | Measure how wrong an observer is about a target (cosine distance: perception vs. reality) |
| **Perception matrix** | How multiple observers see the same target, with pairwise agreement |
| **Arc tracking** | Embedding snapshots over time: drift ranking, arc comparison, moment retrieval |
| **Thematic clustering** | K-means clustering on entity embeddings to reveal emergent story themes |
| **Semantic join** | Cross-field queries like "characters whose desires conflict with Alice's wounds" |
| **What-if analysis** | Preview embedding shift from a character learning a fact — without committing |
| **Unresolved tensions** | Character pairs with high perception asymmetry and few shared scenes — scenes waiting to be written |
| **Knowledge conflicts** | Characters who believe contradictory facts (believes_wrongly detection) |
| **Influence propagation** | Trace how information could spread through the relationship network |
| **Graph analytics** | Centrality metrics (degree, betweenness, closeness) to find structural protagonists and bridge characters |
| **Consistency checking** | Validate entity consistency against universe facts, timeline ordering, and referential integrity |
| **Impact analysis** | Preview which entities would be affected by a change, with severity levels |

### Dual Interface

**CLI**: 30+ commands including high-level workflows (`explore`, `ask`, `find`), graph operations (`path`, `references`), entity management (`create`, `update`, `delete`, `protect`), and organized subcommands: `analyze` (20+ analytics), `world` (status, health, import/export, backfill, validate, graph), `session` (context, pin/unpin).

**MCP Server**: 5 tools with 70 total operations — query (40 ops), mutate (25 ops), session (3 ops), export_world (1 op), generate_graph (1 op). Plus resources (`narra://` URIs) and workflow prompts.

### Technical

- Embedded SurrealDB + RocksDB — no external database to install or manage
- Local BGE-small-en-v1.5 embeddings (384 dimensions, fully offline via fastembed)
- graphrs for graph analytics, linfa for K-means clustering
- Cross-platform: macOS (Intel + Apple Silicon), Linux (x86_64 + aarch64)
- Single statically-linked binary, release builds with LTO
- JSON output mode (`--json`) for scripting and integration

## Quick Start

```bash
# Install (macOS Apple Silicon — see Installation for other platforms)
curl -sL https://github.com/florinutz/narra/releases/latest/download/narra-aarch64-apple-darwin.tar.gz | tar xz
sudo mv narra /usr/local/bin/

# Create a character
narra create character --name "Alice Voss" --role "Detective" \
  --description "Homicide detective haunted by her partner's unsolved murder" \
  --profile '{"wound": ["partner murdered 3 years ago"], "secret": ["knows who did it"]}'

# Generate embeddings, then search by meaning
narra world backfill
narra find --semantic-only "characters hiding dark secrets"
```

## Installation

### From GitHub Releases

Download the latest binary for your platform:

```bash
# macOS (Apple Silicon)
curl -sL https://github.com/florinutz/narra/releases/latest/download/narra-aarch64-apple-darwin.tar.gz | tar xz
sudo mv narra /usr/local/bin/

# macOS (Intel)
curl -sL https://github.com/florinutz/narra/releases/latest/download/narra-x86_64-apple-darwin.tar.gz | tar xz
sudo mv narra /usr/local/bin/

# Linux (x86_64)
curl -sL https://github.com/florinutz/narra/releases/latest/download/narra-x86_64-unknown-linux-gnu.tar.gz | tar xz
sudo mv narra /usr/local/bin/

# Linux (aarch64)
curl -sL https://github.com/florinutz/narra/releases/latest/download/narra-aarch64-unknown-linux-gnu.tar.gz | tar xz
sudo mv narra /usr/local/bin/
```

### From Source

```bash
cargo install --git https://github.com/florinutz/narra.git
```

## CLI Command Reference

Narra's CLI provides intuitive, hierarchical commands for all operations. Commands support `--json` output for scripting and `--brief`/`--full` flags for detail control.

### High-Level Workflows

#### `narra explore <entity>`
Deep entity exploration with relationships, knowledge, perceptions, and semantically similar entities.

```bash
narra explore alice                    # Full exploration
narra explore alice --no-similar       # Skip similarity search
narra explore alice --depth 3          # Extend graph traversal depth
```

#### `narra ask <question>`
Natural language queries about your narrative world.

```bash
narra ask "What secrets does Alice know that Gray doesn't?"
narra ask "Which characters are connected to the murder event?" --limit 20
narra ask "Show me all locations in the city" --context false  # Skip contextual summaries
```

#### `narra find [query]`
Search across entities with automatic hybrid search (keyword + semantic).

```bash
# Basic search (hybrid by default)
narra find "Gray murder"
narra find "betrayal" --type scene --limit 10

# Semantic-only (find by meaning)
narra find --semantic-only "characters struggling with duty"

# Keyword-only (exact text matching)
narra find --keyword-only "Commissioner"

# Advanced search subcommands
narra find join "characters whose desires conflict with Alice's wounds"
narra find knowledge "the royal succession" --character alice
narra find graph alice "betrayal and deception" --hops 2
narra find perspectives "threatening and dangerous" --observer bob
narra find similar alice gray --bias "with more tension"
```

**Find Subcommands:**
- `join` — Cross-type semantic search by meaning
- `knowledge` — Search within character knowledge
- `graph` — Graph proximity + semantic similarity
- `perspectives` — Search perceptions/perspectives by meaning
- `similar` — Find relationships similar to a reference pair

#### `narra path <from> <to>`
Find shortest connection paths between entities.

```bash
narra path alice eddie                 # Shortest paths
narra path alice eddie --max-hops 5    # Extend search depth
narra path alice eddie --include-events  # Include event co-participation
```

#### `narra references <entity>`
Discover what references a target entity.

```bash
narra references alice                 # All references
narra references alice --types scene,event  # Filter by type
narra references location:castle --limit 50
```

### Entity Management

#### `narra create <type>`
Create new entities: `character`, `location`, `event`, `scene`, `knowledge`, `relationship`, `perception`, `fact`, `note`.

```bash
# Character
narra create character --name "Alice Voss" --role "Detective" \
  --description "Homicide detective haunted by her partner's murder" \
  --profile '{"wound": ["partner murdered 3 years ago"], "secret": ["knows who did it"]}'

# Location
narra create location --name "Precinct 13" --loc-type "building" \
  --description "Aging police station" --parent location:downtown

# Event
narra create event --title "The Confrontation" --sequence 10 \
  --description "Alice confronts Gray with evidence" \
  --date "2024-06-15" --date-precision day

# Scene
narra create scene --title "Gray's Office" --event event:confrontation \
  --location location:city_hall --summary "The truth comes out"

# Knowledge
narra create knowledge --character alice --fact "Gray ordered the hit" \
  --certainty knows --method discovered --event event:investigation

# Relationship
narra create relationship --from alice --to gray --type antagonistic \
  --label "Hunter and prey — neither knows who is which"

# Perception
narra create perception --observer bob --target alice \
  --perception "Sees her as relentless and dangerous" \
  --feelings "fear, respect" --tension 8

# Fact (universe rules)
narra create fact --title "Magic requires sacrifice" \
  --description "All magic extracts a personal cost" \
  --categories physics_magic --enforcement strict

# Note
narra create note --title "Plot thread" --body "Revisit Eddie's backstory" \
  --attach-to character:eddie,event:tip
```

#### `narra get <entity>`
Retrieve any entity by ID or name (auto-resolves).

```bash
narra get alice                        # By name (auto-resolves)
narra get character:alice              # By full ID
narra get event:confrontation
```

#### `narra list <type>`
List entities with optional filters.

```bash
narra list character
narra list event --limit 50
narra list knowledge --character alice
narra list fact --category physics_magic --enforcement strict
narra list note --entity character:alice
```

#### `narra update <entity>`
Update entity fields.

```bash
# Update with JSON object
narra update character:alice --fields '{"role": "Lead Detective"}'

# Update individual fields
narra update character:alice --set role="Lead Detective" --set description="..."

# Link/unlink facts
narra update character:alice --link fact:magic_rules
narra update character:alice --unlink fact:old_rule
```

#### `narra delete <entity>`
Delete entity (with impact analysis prompt).

```bash
narra delete character:bob             # Soft delete (respects protection)
narra delete character:bob --hard      # Hard delete (bypass protection)
```

### Entity Protection

#### `narra protect <entity>`
Mark entity as protected. Protected entities trigger CRITICAL severity in impact analysis and require `--hard` flag for deletion.

```bash
narra protect alice                    # Protect protagonist
narra protect event:climax             # Protect key plot point
narra protect location:headquarters    # Protect central location
```

Use cases:
- **Protagonists**: Prevent accidental deletion of main characters
- **Plot anchors**: Protect pivotal events that the story depends on
- **World foundations**: Protect core locations or universe facts

#### `narra unprotect <entity>`
Remove protection from entity.

```bash
narra unprotect character:minor_npc
```

### Analysis Commands

All under `narra analyze <operation>`:

```bash
# Network analysis
narra analyze centrality               # Network centrality metrics
narra analyze centrality --scope character:alice --limit 20
narra analyze influence alice --depth 3  # Influence propagation

# Knowledge analysis
narra analyze irony                    # Dramatic irony report
narra analyze irony --character alice --threshold 5
narra analyze asymmetries alice gray   # Pairwise knowledge gaps
narra analyze conflicts                # Knowledge conflicts (believes_wrongly)
narra analyze conflicts --character alice

# Perception analysis
narra analyze perception-gap alice bob  # How wrong is alice about bob?
narra analyze perception-matrix bob    # How do all observers see bob?
narra analyze perception-shift alice bob  # How has alice's view evolved?
narra analyze tensions                 # Unresolved perception tensions

# Arc tracking (requires baseline snapshots)
narra analyze arc-drift                # Most-changed entities
narra analyze arc-drift --type character --limit 20
narra analyze arc-history alice        # Entity evolution timeline
narra analyze arc-compare alice bob --window "recent:10"
narra analyze arc-moment alice --event event:betrayal

# Thematic analysis
narra analyze themes                   # K-means clustering
narra analyze themes --types character,event --clusters 5
narra analyze thematic-gaps --min-size 3

# Temporal & consistency
narra analyze temporal alice --event event:confrontation
narra analyze contradictions alice --depth 3
narra analyze impact alice --description "major personality shift"

# Composite reports
narra analyze situation-report         # High-level narrative overview
narra analyze dossier alice           # Comprehensive character report
narra analyze scene-prep alice,bob,gray  # Scene planning for character meeting
narra analyze what-if alice --fact knowledge:secret --certainty suspects
```

### Session Management

Session state persists between CLI invocations and MCP server usage.

#### `narra session context`
View current session state: hot entities (recently accessed), pinned entities, pending decisions, and world overview.

```bash
narra session context
narra session context --json           # Machine-readable output
```

Session context includes:
- **Hot entities**: Recently viewed or modified (last 50)
- **Pinned entities**: Manually pinned for persistent reference
- **Pending decisions**: Entities with validation warnings
- **World overview**: Entity counts, embedding coverage

#### `narra session pin <entity>`
Pin entity to persistent session context.

```bash
narra session pin alice                # Keep protagonist in context
narra session pin event:climax         # Pin pivotal event
narra session pin location:headquarters
```

Use cases:
- **Active writing**: Pin characters/locations you're currently writing
- **Continuity**: Pin entities you need to reference frequently
- **Cross-session**: Maintain context across MCP and CLI usage

#### `narra session unpin <entity>`
Remove entity from pinned context.

```bash
narra session unpin character:minor_npc
```

### World Management

All under `narra world <operation>`:

#### `narra world status`
World overview dashboard with entity counts and embedding coverage.

```bash
narra world status
```

#### `narra world health`
Embedding health report: coverage, staleness, missing embeddings.

```bash
narra world health
```

#### `narra world backfill`
Generate embeddings for all entities (run after initial data entry).

```bash
narra world backfill                   # All entity types
narra world backfill --type character  # Single type only
```

#### `narra world baseline-arcs`
Create baseline arc snapshots for arc tracking (run once after backfill).

```bash
narra world baseline-arcs              # All types with embeddings
narra world baseline-arcs --type character
```

#### `narra world import <file>`
Import world data from YAML file.

```bash
narra world import story-world.yaml
narra world import story-world.yaml --on-conflict update  # Merge with existing
narra world import story-world.yaml --dry-run  # Preview without writing
```

Conflict modes: `error` (default, skip conflicts), `skip` (silent), `update` (merge fields).

#### `narra world export`
Export world data to YAML.

```bash
narra world export                     # Auto-named export
narra world export -o backup.yaml      # Custom filename
```

#### `narra world validate`
Validate entity consistency against universe facts and timeline.

```bash
narra world validate                   # General check
narra world validate character:alice   # Single entity
```

#### `narra world graph`
Generate Mermaid relationship diagram.

```bash
narra world graph --scope full         # All characters
narra world graph --scope character:alice --depth 2
narra world graph -o graph.mmd         # Save to file
```

### Batch Operations

#### `narra batch <type>`
Batch-create entities from YAML (stdin or file).

```bash
# From file
narra batch character --file characters.yaml

# From stdin
cat characters.yaml | narra batch character
```

### Output Control

Global flags available on all commands:

```bash
--json           # JSON output for scripting
--md             # Markdown output
--brief          # Less detail
--full           # Maximum detail
--no-semantic    # Disable semantic search (use keyword only)
```

### Shell Completions

Generate shell completions for your shell:

```bash
narra completions bash > /etc/bash_completion.d/narra
narra completions zsh > ~/.zsh/completion/_narra
narra completions fish > ~/.config/fish/completions/narra.fish
```

### Claude Code Plugin

For Claude Code integration with slash commands, skills, and hooks for fiction writing workflows:

```
https://github.com/florinutz/flo-market/tree/main/plugins/narra
```

The plugin auto-downloads the binary, provides session management, and exposes Narra operations as conversational commands.

## CLI Walkthrough

A noir mystery walkthrough: detective Alice Voss, corrupt commissioner Gray, and street informant Eddie — building a story world from scratch.

### 1. Bootstrap the world

The fastest way to set up a story world is via YAML import, which lets you specify your own IDs:

```bash
narra world import noir-world.yaml
```

<details>
<summary>noir-world.yaml</summary>

```yaml
characters:
  - id: alice
    name: "Alice Voss"
    role: "Detective"
    description: "Homicide detective, 15 years on the force. Methodical, relentless."
    profile:
      wound: ["partner murdered 3 years ago"]
      desire: ["justice for her partner"]
      secret: ["knows Commissioner Gray ordered the hit"]
      contradiction: ["upholds the law while planning extralegal revenge"]

  - id: gray
    name: "Commissioner Gray"
    role: "Antagonist"
    description: "Police commissioner with a public image of reform. Privately runs a protection racket."
    profile:
      wound: ["grew up in poverty, fears losing power"]
      desire: ["absolute control of the city"]
      secret: ["ordered the murder of Alice's partner"]
      contradiction: ["champion of justice who is the biggest criminal"]

  - id: eddie
    name: "Eddie Malone"
    role: "Informant"
    description: "Street-level informant who plays all sides. Knows more than he lets on."
    profile:
      wound: ["abandoned by family at 14"]
      desire: ["enough money to disappear"]
      secret: ["suspects Gray is dirty but has no proof"]
      contradiction: ["sells information to survive while hating betrayal"]

locations:
  - id: precinct
    name: "Precinct 13"
    loc_type: "building"
    description: "Aging police station in the warehouse district. Alice's home base."

  - id: blind_pig
    name: "The Blind Pig"
    loc_type: "bar"
    description: "Dive bar where Eddie brokers information. Back booth, cash only."

  - id: city_hall
    name: "City Hall"
    loc_type: "building"
    description: "Commissioner Gray's seat of power. Marble corridors, closed doors."

events:
  - id: murder
    title: "Partner's Murder"
    sequence: 1
    description: "Alice's partner killed in an apparent robbery. Case goes cold."

  - id: promotion
    title: "Gray's Promotion"
    sequence: 2
    description: "Gray becomes commissioner, promising to clean up the city."

  - id: tip
    title: "Eddie's Tip"
    sequence: 3
    description: "Eddie approaches Alice with a rumor about Gray's protection racket."

  - id: confrontation
    title: "The Confrontation"
    sequence: 4
    description: "Alice confronts Gray with evidence. The truth comes out."

relationships:
  - from_character_id: "character:alice"
    to_character_id: "character:gray"
    rel_type: "antagonistic"
    label: "Hunter and prey — neither knows who is which"

  - from_character_id: "character:eddie"
    to_character_id: "character:alice"
    rel_type: "professional"
    subtype: "informant"
    label: "Eddie feeds Alice information for cash"

  - from_character_id: "character:gray"
    to_character_id: "character:eddie"
    rel_type: "professional"
    subtype: "asset"
    label: "Gray uses Eddie as eyes on the street"

knowledge:
  - character_id: "character:alice"
    target_id: "character:gray"
    fact: "Commissioner Gray ordered the murder of Alice's partner"
    certainty: "knows"
    method: "discovered"

  - character_id: "character:eddie"
    target_id: "character:gray"
    fact: "Commissioner Gray is connected to organized crime"
    certainty: "suspects"
    method: "deduced"

  - character_id: "character:gray"
    target_id: "character:alice"
    fact: "No one suspects my involvement in the partner's murder"
    certainty: "believes_wrongly"
    method: "deduced"
```

</details>

> Three characters with intertwined secrets. Profile keys (wound, desire, secret, contradiction) become part of each character's embedding — enabling semantic search across psychological dimensions. Alice *knows* what Gray *believes no one knows*. Eddie *suspects* but can't prove it.

All entity creation uses the unified `narra create <type>` interface. See the [CLI Command Reference](#cli-command-reference) below for complete command documentation.

### 2. Generate embeddings

```bash
narra world backfill
```

> Computes BGE-small-en-v1.5 embeddings for all entities. This enables semantic search, arc tracking, and perception analysis.

### 3. Explore and search

```bash
# Deep exploration: relationships, knowledge, perceptions, similar entities
narra explore alice

# Natural language questions
narra ask "What secrets does Alice know that Gray doesn't?"

# Simple search (hybrid by default - combines keyword + semantic)
narra find "Gray murder"

# Semantic-only search (find by meaning, not keywords)
narra find --semantic-only "characters hiding dark secrets"

# Advanced searches
narra find join "characters whose desires conflict with Alice's wounds"
narra find knowledge "the royal succession" --character alice
narra find graph alice "betrayal and deception" --hops 2
narra find perspectives "threatening and dangerous" --observer bob
narra find similar alice gray  # Find relationships similar to Alice-Gray dynamic

# Connection analysis
narra path alice eddie  # Shortest connection paths
narra references alice  # What mentions/references Alice?
```

> High-level commands (`explore`, `ask`, `find`) provide intuitive access to Narra's intelligence. Semantic search finds entities by psychological profile, not just keywords. Hybrid search combines both approaches.

### 4. Analyze the network

```bash
# Who are the most connected characters?
narra analyze centrality

# How would information spread from Eddie?
narra analyze influence eddie

# What knowledge asymmetries exist?
narra analyze irony

# What does Alice know that Gray doesn't, and vice versa?
narra analyze asymmetries alice gray

# Who believes things that are wrong?
narra analyze conflicts

# Where are the unresolved tensions?
narra analyze tensions

# Which characters have changed the most?
narra analyze arc-drift
```

### 5. Visualize

```bash
narra world graph --scope full
```

> Generates a Mermaid diagram of the relationship network.

### 6. Export

```bash
narra world export -o noir-export.yaml
```

> Exports the entire world to YAML. Re-importable with `narra world import noir-export.yaml`.

With the MCP server, an AI assistant can run all of these operations plus many more — including perception gaps, perception matrices, arc comparison, thematic clustering, what-if analysis, and semantic graph search.

## MCP Server

Start the MCP server (stdio transport, designed for Claude Code and other MCP clients):

```bash
narra mcp
```

### Tools Overview

| Tool | Operations | Purpose |
|---|---|---|
| **query** | 40 | Read-only: search, analytics, arc tracking, perception analysis, validation, impact preview |
| **mutate** | 25 | Write: create/update/delete entities, record knowledge, batch ops, import, protect entities |
| **session** | 3 | Context management: hot entities, pinned items, pending decisions |
| **export_world** | 1 | Export world to YAML (NarraImport-compatible, re-importable) |
| **generate_graph** | 1 | Generate Mermaid relationship diagram to `.planning/exports/` |

### Highlighted Operations

| Operation | What it does |
|---|---|
| `DramaticIronyReport` | Knowledge asymmetries with scene-count since asymmetry arose |
| `KnowledgeAsymmetries` | Pairwise knowledge gaps between two characters with tension context |
| `PerceptionGap` | Cosine distance between observer's view and target's reality |
| `PerceptionMatrix` | How multiple observers see one target, with pairwise agreement |
| `PerceptionShift` | Track how an observer's view evolved over time via arc snapshots |
| `ArcDrift` | Rank entities by total embedding drift — who has changed most |
| `ArcHistory` | View embedding snapshot timeline for an entity's evolution |
| `ArcComparison` | Compare two entity trajectories for convergence/divergence |
| `ArcMoment` | Get entity state at a specific event (nearest snapshot) |
| `SemanticJoin` | Cross-field semantic queries across entity types |
| `SemanticKnowledge` | Search character knowledge by meaning, optionally per-character |
| `SemanticGraphSearch` | Graph + vector: find connected entities matching a concept |
| `PerspectiveSearch` | Search perspective embeddings semantically |
| `SimilarRelationships` | Find relationships similar to a reference pair |
| `WhatIf` | Preview impact of a character learning a fact — without committing |
| `UnresolvedTensions` | High-asymmetry pairs with few shared scenes — scenes waiting to be written |
| `ThematicClustering` | K-means on embeddings to discover emergent story themes |
| `ThematicGaps` | Find clusters missing expected entity types |
| `InfluencePropagation` | Trace how information could spread through the network |
| `KnowledgeConflicts` | Find characters who believe contradictory facts |
| `SituationReport` | High-level narrative overview: irony, conflicts, tensions, themes |
| `CharacterDossier` | Comprehensive character report: network, knowledge, influence, perceptions |
| `ScenePlanning` | Scene prep for character meetings: dynamics, irony, tensions |
| `AnalyzeImpact` | Preview change impact before mutation with affected entity severity |
| `ValidateEntity` | Check entity consistency against facts, timeline, relationships |
| `InvestigateContradictions` | Graph traversal to find what contradicts an entity |

### Resources and Prompts

**Resources** (`narra://` URIs):
- `narra://session/context` — hot entities, pinned items, pending decisions
- `narra://entity/{type}:{id}` — full entity view with attributes and relationships
- `narra://consistency/issues` — current violations by severity
- `narra://schema/import-template` — YAML template for world import
- `narra://schema/import-schema` — JSON Schema for import validation

**Prompts**:
- `check_consistency` — guided validation with fix suggestions
- `dramatic_irony` — knowledge asymmetry analysis between characters

### Claude Code Plugin

For integrated Claude Code workflows, install the [Narra plugin](https://github.com/florinutz/flo-market/tree/main/plugins/narra). It provides slash commands, skills, hooks, and automatic binary management.

## Data Storage

By default, Narra stores data in `~/.narra/`. You can override this per-project or globally:

```bash
# Per-project: create a .narra directory in your project root
mkdir .narra
narra list character  # uses ./.narra/ automatically

# Environment variable
export NARRA_DATA_PATH=/path/to/data

# CLI flag (highest priority)
narra --data-path /path/to/data list character
```

Resolution order: `--data-path` flag > `NARRA_DATA_PATH` env > `./.narra` (if exists) > `~/.narra`

## Architecture

```
CLI (clap) ──┐
             ├─→ AppContext ──→ Services ──→ Repositories ──→ SurrealDB (RocksDB)
MCP (rmcp) ──┘
```

- **AppContext** (`init.rs`) — central dependency wiring. Both CLI and MCP share the same services
- **Trait objects** — all services use `Arc<dyn XService + Send + Sync>` for testability
- **Tagged enum dispatch** — MCP tools receive `#[serde(tag = "operation")]` discriminated unions
- **Embedding service trait** — `EmbeddingService` with `LocalEmbeddingService` (fastembed) for production and `NoopEmbeddingService` for tests
- **Caching** — moka async caches for context and summary services
- **Data providers** — services like IronyService, InfluenceService, GraphAnalyticsService use data provider traits, enabling unit tests with mock data

## Development

```bash
git clone https://github.com/florinutz/narra.git
cd narra

cargo build                      # Debug build
cargo test                       # All tests
cargo test --lib                 # Unit tests only
cargo test --test '*'            # Integration tests only
cargo clippy -- -D warnings      # Lint (CI enforces warnings-as-errors)
cargo fmt -- --check             # Format check
```

Key make targets:

```bash
make build          # Debug build
make lint           # Clippy + fmt check
make test           # All tests
make quick-check    # Lint + build (no tests)
make test-embedding # Embedding-specific tests
```

Tests use an isolated SurrealDB instance per test (temp directory), with fluent builders for test data (`CharacterBuilder`, `LocationBuilder`, etc.) in `tests/common/`.

## License

[MIT](LICENSE)
