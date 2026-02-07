-- Narra Phase 16 Schema Extension
-- Embedding infrastructure: vector fields and staleness tracking

-- ==============================================================================
-- CHARACTER EMBEDDING FIELDS
-- ==============================================================================
DEFINE FIELD embedding ON character TYPE option<array<float>> DEFAULT NONE;
DEFINE FIELD embedding_stale ON character TYPE bool DEFAULT true;

-- ==============================================================================
-- LOCATION EMBEDDING FIELDS
-- ==============================================================================
DEFINE FIELD embedding ON location TYPE option<array<float>> DEFAULT NONE;
DEFINE FIELD embedding_stale ON location TYPE bool DEFAULT true;

-- ==============================================================================
-- EVENT EMBEDDING FIELDS
-- ==============================================================================
DEFINE FIELD embedding ON event TYPE option<array<float>> DEFAULT NONE;
DEFINE FIELD embedding_stale ON event TYPE bool DEFAULT true;

-- ==============================================================================
-- SCENE EMBEDDING FIELDS
-- ==============================================================================
DEFINE FIELD embedding ON scene TYPE option<array<float>> DEFAULT NONE;
DEFINE FIELD embedding_stale ON scene TYPE bool DEFAULT true;
