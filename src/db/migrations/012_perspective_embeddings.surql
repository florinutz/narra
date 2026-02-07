-- Perspective Embeddings: per-observer subjective view of target characters.
-- Stored on perceives edges. Computed from perception data + observer's knowledge + shared scenes.

DEFINE FIELD embedding ON perceives TYPE option<array<float>> DEFAULT NONE;
DEFINE FIELD embedding_stale ON perceives TYPE bool DEFAULT true;

-- Extend arc_snapshot to support perspective embedding tracking.
-- Redefine entity_type to accept "perspective" alongside existing types.
DEFINE FIELD entity_type ON arc_snapshot TYPE string
    ASSERT $value IN ["character", "knowledge", "perspective"];
