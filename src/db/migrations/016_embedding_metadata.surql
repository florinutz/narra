-- Embedding metadata: tracks active embedding model info per world.
-- Used to detect model mismatches after provider changes.

DEFINE TABLE IF NOT EXISTS world_meta SCHEMAFULL;

DEFINE FIELD IF NOT EXISTS embedding_model ON TABLE world_meta TYPE option<string>;
DEFINE FIELD IF NOT EXISTS embedding_dimensions ON TABLE world_meta TYPE option<int>;
DEFINE FIELD IF NOT EXISTS embedding_provider ON TABLE world_meta TYPE option<string>;
DEFINE FIELD IF NOT EXISTS last_backfill_at ON TABLE world_meta TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS updated_at ON TABLE world_meta TYPE option<datetime>
    DEFAULT time::now();
