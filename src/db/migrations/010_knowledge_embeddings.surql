-- Knowledge Embeddings
-- Adds embedding fields and HNSW vector index to knowledge table for semantic search.
-- BGE-small-en-v1.5 produces 384-dimensional F32 embeddings.

-- ==============================================================================
-- KNOWLEDGE EMBEDDING FIELDS
-- ==============================================================================
DEFINE FIELD embedding ON knowledge TYPE option<array<float>> DEFAULT NONE;
DEFINE FIELD embedding_stale ON knowledge TYPE bool DEFAULT true;

-- ==============================================================================
-- KNOWLEDGE EMBEDDING HNSW INDEX
-- ==============================================================================
DEFINE INDEX knowledge_embedding_hnsw ON knowledge FIELDS embedding HNSW DIMENSION 384 DIST COSINE TYPE F32;
