-- Character facets: multi-vector embeddings for faceted search.
-- Provides separate embeddings for identity, psychology, social, and narrative dimensions.

-- Identity facet: name, roles, aliases
DEFINE FIELD IF NOT EXISTS identity_embedding ON TABLE character TYPE option<array<float>> DEFAULT NONE;
DEFINE FIELD IF NOT EXISTS identity_composite ON TABLE character TYPE option<string> DEFAULT NONE;
DEFINE FIELD IF NOT EXISTS identity_stale ON TABLE character TYPE bool DEFAULT false;

-- Psychology facet: profile categories
DEFINE FIELD IF NOT EXISTS psychology_embedding ON TABLE character TYPE option<array<float>> DEFAULT NONE;
DEFINE FIELD IF NOT EXISTS psychology_composite ON TABLE character TYPE option<string> DEFAULT NONE;
DEFINE FIELD IF NOT EXISTS psychology_stale ON TABLE character TYPE bool DEFAULT false;

-- Social facet: relationships + inbound perceptions
DEFINE FIELD IF NOT EXISTS social_embedding ON TABLE character TYPE option<array<float>> DEFAULT NONE;
DEFINE FIELD IF NOT EXISTS social_composite ON TABLE character TYPE option<string> DEFAULT NONE;
DEFINE FIELD IF NOT EXISTS social_stale ON TABLE character TYPE bool DEFAULT false;

-- Narrative facet: scene participation + knowledge
DEFINE FIELD IF NOT EXISTS narrative_embedding ON TABLE character TYPE option<array<float>> DEFAULT NONE;
DEFINE FIELD IF NOT EXISTS narrative_composite ON TABLE character TYPE option<string> DEFAULT NONE;
DEFINE FIELD IF NOT EXISTS narrative_stale ON TABLE character TYPE bool DEFAULT false;

-- Per-facet arc snapshots: optional field to distinguish facet-specific snapshots
DEFINE FIELD IF NOT EXISTS facet ON TABLE arc_snapshot TYPE option<string> DEFAULT NONE;
