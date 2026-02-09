# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Narra

Narrative intelligence engine for AI-assisted fiction writing. Dual-mode Rust binary: CLI tool and MCP server (via `rmcp`). Manages story world state — characters, locations, events, scenes, relationships, knowledge, perceptions — in SurrealDB with RocksDB persistence. Provides semantic search (BGE-small-en-v1.5 via `fastembed`), arc tracking, perspective vectors, and graph analytics.

## Build & Test Commands

```bash
cargo build                      # Debug build
cargo build --release            # Release build (LTO, stripped)
cargo test                       # All tests (unit + integration)
cargo test --lib                 # Unit tests only
cargo test --test '*'            # Integration tests only
cargo test <test_name>           # Single test by name
cargo insta test --review        # Update and review insta snapshots
cargo clippy -- -D warnings      # Lint (CI enforces warnings-as-errors)
cargo fmt                        # Format
cargo fmt -- --check             # Format check (CI)
```

The `Makefile` wraps these as `make build`, `make test`, `make lint`, etc.

## Architecture

### Layered Design

```
main.rs → CLI (clap)  ──┐
                         ├→ AppContext (init.rs) → Services → Repositories → SurrealDB
MCP Server (rmcp)    ──┘
```

**`init.rs` / `AppContext`**: Central dependency wiring. Both CLI and MCP server create an `AppContext` that holds `Arc`-wrapped services and repositories. All services use trait objects (`Arc<dyn XService + Send + Sync>`) for testability.

**Data path resolution**: explicit `--data-path` > `NARRA_DATA_PATH` env > `./.narra` (if exists) > `~/.narra`

### Key Modules

- **`models/`** — Serde structs for entities (Character, Location, Event, Scene, Relationship, Knowledge, Perception, Fact, Note) with `*Create`/`*Update` variants
- **`repository/`** — `SurrealEntityRepository`, `SurrealKnowledgeRepository`, `SurrealRelationshipRepository` — direct SurrealDB queries
- **`services/`** — Business logic: search (keyword/semantic/hybrid), consistency checking, impact analysis, graph analytics, influence propagation, irony detection, clustering (linfa), context/summary with moka caching
- **`embedding/`** — `EmbeddingService` trait with `LocalEmbeddingService` (fastembed BGE-small-en-v1.5, 384 dims) and `NoopEmbeddingService` for tests. `StalenessManager` tracks embedding freshness. `BackfillService` for batch embedding generation
- **`mcp/`** — MCP server using `rmcp` crate. 5 consolidated tools (query, mutate, session, export_world, generate_graph). Tool definitions are in `server.rs` via `#[tool]` macros; implementations in `tools/*.rs`. Also has resources and prompts
- **`mcp/types.rs`** — Request/response enums using `#[serde(tag = "operation")]` discriminated unions. `QueryRequest` has 40 variants, `MutationRequest` has 25 variants
- **`cli/`** — Clap-derive command tree (`Commands` enum with subcommands). Handlers in `cli/handlers/` dispatch to services
- **`session/`** — Session state persistence (hot entities, pinned entities, pending decisions) via JSON file
- **`db/`** — SurrealDB connection (`RocksDb` engine, namespace `narra`, database `world`) and schema migrations (`.surql` files applied in order)

### Database

SurrealDB with RocksDB backend. Schema is applied via numbered `.surql` migration files in `src/db/migrations/`. Entity IDs use SurrealDB's `table:id` format (e.g., `character:alice`). Graph edges use SurrealDB's `RELATE` syntax for relationships, knowledge, and perceptions.

### MCP Tool Pattern

The MCP server consolidates operations into 5 tools. Each tool receives a tagged enum request:
```rust
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum QueryRequest { Lookup { ... }, Search { ... }, ... }
```
Tool handler implementations are split across `mcp/tools/*.rs` files (query.rs, mutate.rs, validate.rs, etc.) but the `#[tool]` macro declarations must remain in the `#[tool_router] impl NarraServer` block in `server.rs`.

## CLI Command Structure

Modern unified interface (no legacy commands):

**High-level workflows:**
- `narra explore <entity>` — Deep entity exploration
- `narra ask <question>` — Natural language queries
- `narra find [query]` — Hybrid search with subcommands: `join`, `knowledge`, `graph`, `perspectives`, `similar`
- `narra path <from> <to>` — Connection paths
- `narra references <entity>` — What references an entity

**Entity management:**
- `narra create <type>` — Create: character, location, event, scene, knowledge, relationship, perception, fact, note
- `narra get <entity>` — Retrieve by ID or name
- `narra list <type>` — List with filters
- `narra update <entity>` — Update fields, link/unlink
- `narra delete <entity>` — Delete (respects protection)
- `narra protect/unprotect <entity>` — Entity protection

**Analysis:**
- `narra analyze <operation>` — 20+ operations: centrality, influence, irony, asymmetries, conflicts, tensions, arc-drift, arc-history, arc-compare, arc-moment, perception-gap, perception-matrix, perception-shift, themes, thematic-gaps, temporal, contradictions, what-if, impact, situation-report, dossier, scene-prep

**World management:**
- `narra world status/health` — Overview and diagnostics
- `narra world backfill/baseline-arcs` — Embeddings and arc tracking setup
- `narra world import/export` — YAML round-trip
- `narra world validate` — Consistency checking
- `narra world graph` — Mermaid diagram generation

**Session:**
- `narra session context` — View session state
- `narra session pin/unpin <entity>` — Persistent context management

**Global flags:**
- `--json` — JSON output
- `--md` — Markdown output
- `--brief/--full` — Detail control
- `--no-semantic` — Disable semantic search

## Testing

- **Test harness**: `tests/common/harness.rs` — `TestHarness::new()` creates an isolated SurrealDB instance in a temp directory per test. Use `test_embedding_service()` for a no-op embedding service
- **Builders**: `tests/common/builders.rs` — Fluent builders (`CharacterBuilder`, `LocationBuilder`, `EventBuilder`, `SceneBuilder`, `KnowledgeBuilder`) for test data
- **Snapshots**: Uses `insta` crate for snapshot testing (YAML format). Snapshots stored in `tests/handlers/snapshots/`
- **Assertions**: Uses `pretty_assertions` for readable diffs
- Integration tests are in `tests/` as separate files; each async test creates its own `TestHarness`

## Lint Configuration

`Cargo.toml` allows `clippy::redundant_closure_for_method_calls`. CI runs `cargo clippy -- -D warnings` and `cargo fmt -- --check`.

## Workflow Rules

- Use `gh` CLI for all GitHub operations (PRs, issues, action runs, etc.)
- Never commit or push — leave that to me
- Before considering any work item done, run the appropriate tests based on change scope:
  - Formatting/lint-only changes → `make quick-check`
  - Code changes that could affect tests → `make quick-check` + `cargo test`
  - Embedding/vector/semantic code touched → also `make test-embedding`

Don't be shy to notice problems and propose improvements. Keep an eye on unexplored lanes of flight and possibilities. Be proactive. Use menus to drill down or show proposals or options to me.
As you work on it, always keep an eye on possible improvements, on what could be changed, on new possibilities, on possibly unrelated (to the task at hand) problems you might notice around, and signal to me if you're blocked on anything.
Be proactive, propose stuff, be creative, think about the bigger picture, use menus (e.g. options, multiple or single, free input, whatever) to drill down or to expand my thought process or cover my gaps when you became aware of them.

Always proactively use (context-related) menus to both drill down with the user, and to propose directions to cover gaps, expand the thinking, or to explore new possibilities with the user.
