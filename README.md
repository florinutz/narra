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

**CLI**: 20+ commands including `analyze` subcommands (centrality, influence, irony, asymmetries, conflicts, tensions, arc-drift), entity management, search, import/export, graph visualization, and embedding management.

**MCP Server**: 5 tools with 67 total operations — query (37 ops), mutate (25 ops), session (3 ops), export_world (1 op), generate_graph (1 op). Plus resources (`narra://` URIs) and workflow prompts.

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
narra character create --name "Alice Voss" --role "Detective" \
  --description "Homicide detective haunted by her partner's unsolved murder" \
  --profile '{"wound": ["partner murdered 3 years ago"], "secret": ["knows who did it"]}'

# Generate embeddings, then search by meaning
narra backfill
narra search --semantic "characters hiding dark secrets"
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
narra import noir-world.yaml
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

Individual CLI commands are also available (`narra character create`, `narra relationship create`, `narra knowledge record`, etc.) — see `narra --help` for the full command tree.

### 2. Generate embeddings

```bash
narra backfill
```

> Computes BGE-small-en-v1.5 embeddings for all entities. This enables semantic search, arc tracking, and perception analysis.

### 3. Search by meaning

```bash
narra search --semantic "characters hiding dark secrets"
narra search --semantic "places where deals are made"
narra search --hybrid "Gray murder"
```

> Semantic search finds characters by psychological profile, not just keywords. Hybrid search combines both approaches.

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
narra graph --scope full
```

> Generates a Mermaid diagram of the relationship network.

### 6. Export

```bash
narra export -o noir-export.yaml
```

> Exports the entire world to YAML. Re-importable with `narra import noir-export.yaml`.

With the MCP server, an AI assistant can run all of these operations plus many more — including perception gaps, perception matrices, arc comparison, thematic clustering, what-if analysis, and semantic graph search.

## MCP Server

Start the MCP server (stdio transport, designed for Claude Code and other MCP clients):

```bash
narra mcp
```

### Tools Overview

| Tool | Operations | Purpose |
|---|---|---|
| **query** | 37 | Read-only: search, analytics, arc tracking, perception analysis, validation |
| **mutate** | 25 | Write: create/update/delete entities, record knowledge, batch ops, import |
| **session** | 3 | Context management: hot entities, pinned items |
| **export_world** | 1 | Export world to YAML |
| **generate_graph** | 1 | Generate Mermaid relationship diagram |

### Highlighted Operations

| Operation | What it does |
|---|---|
| `DramaticIronyReport` | Knowledge asymmetries with scene-count since asymmetry arose |
| `PerceptionGap` | Cosine distance between observer's view and target's reality |
| `PerceptionMatrix` | How multiple observers see one target, with pairwise agreement |
| `ArcDrift` | Rank entities by total embedding drift — who has changed most |
| `SemanticJoin` | Cross-field semantic queries across entity types |
| `WhatIf` | Preview impact of a character learning a fact — without committing |
| `UnresolvedTensions` | High-asymmetry pairs with few shared scenes — scenes waiting to be written |
| `ThematicClustering` | K-means on embeddings to discover emergent story themes |
| `InfluencePropagation` | Trace how information could spread through the network |
| `KnowledgeConflicts` | Find characters who believe contradictory facts |

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
narra character list  # uses ./.narra/ automatically

# Environment variable
export NARRA_DATA_PATH=/path/to/data

# CLI flag (highest priority)
narra --data-path /path/to/data character list
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
