-- HNSW Vector Indexes for Semantic Search
-- Replaces brute-force cosine similarity scan with approximate nearest-neighbor via HNSW.
-- BGE-small-en-v1.5 produces 384-dimensional F32 embeddings.

-- ==============================================================================
-- CHARACTER EMBEDDING INDEX
-- ==============================================================================
DEFINE INDEX character_embedding_hnsw ON character FIELDS embedding HNSW DIMENSION 384 DIST COSINE TYPE F32;

-- ==============================================================================
-- LOCATION EMBEDDING INDEX
-- ==============================================================================
DEFINE INDEX location_embedding_hnsw ON location FIELDS embedding HNSW DIMENSION 384 DIST COSINE TYPE F32;

-- ==============================================================================
-- EVENT EMBEDDING INDEX
-- ==============================================================================
DEFINE INDEX event_embedding_hnsw ON event FIELDS embedding HNSW DIMENSION 384 DIST COSINE TYPE F32;

-- ==============================================================================
-- SCENE EMBEDDING INDEX
-- ==============================================================================
DEFINE INDEX scene_embedding_hnsw ON scene FIELDS embedding HNSW DIMENSION 384 DIST COSINE TYPE F32;
