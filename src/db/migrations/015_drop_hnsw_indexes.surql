-- Drop unused HNSW vector indexes (defined in 009_hnsw_vector_indexes.surql).
-- These indexes are never queried — semantic search uses brute-force cosine
-- similarity which is reliable in SurrealDB embedded RocksDB mode and fast
-- enough for narrative data scales.  Removing them eliminates write amplification
-- on every embedding update.

REMOVE INDEX IF EXISTS character_embedding_hnsw ON TABLE character;
REMOVE INDEX IF EXISTS location_embedding_hnsw ON TABLE location;
REMOVE INDEX IF EXISTS event_embedding_hnsw ON TABLE event;
REMOVE INDEX IF EXISTS scene_embedding_hnsw ON TABLE scene;
