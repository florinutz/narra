-- Migration 013: Embedding infrastructure improvements
-- relates_to edge embeddings + composite_text storage for debugging/no-op detection

-- relates_to edge embeddings (like perceives, but for structural relationships)
DEFINE FIELD embedding ON relates_to TYPE option<array<float>> DEFAULT NONE;
DEFINE FIELD embedding_stale ON relates_to TYPE bool DEFAULT true;

-- Composite text storage on all embeddable tables
-- Stores the text that was last embedded, for debugging and no-op detection
DEFINE FIELD composite_text ON character TYPE option<string> DEFAULT NONE;
DEFINE FIELD composite_text ON location TYPE option<string> DEFAULT NONE;
DEFINE FIELD composite_text ON event TYPE option<string> DEFAULT NONE;
DEFINE FIELD composite_text ON scene TYPE option<string> DEFAULT NONE;
DEFINE FIELD composite_text ON knowledge TYPE option<string> DEFAULT NONE;
DEFINE FIELD composite_text ON perceives TYPE option<string> DEFAULT NONE;
DEFINE FIELD composite_text ON relates_to TYPE option<string> DEFAULT NONE;
