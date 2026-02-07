-- Narra Phase 4 Schema Extension
-- Search Infrastructure: Full-text search indexes with BM25 ranking

-- ==============================================================================
-- FULL-TEXT SEARCH ANALYZER
-- Uses ascii normalization, lowercase, and English stemming for search
-- ==============================================================================

DEFINE ANALYZER narra_analyzer TOKENIZERS class FILTERS ascii, lowercase, snowball(english);

-- ==============================================================================
-- CHARACTER SEARCH INDEXES
-- Full-text search on character names for SRCH-01, SRCH-02
-- ==============================================================================

DEFINE INDEX character_name_ft ON character FIELDS name SEARCH ANALYZER narra_analyzer BM25;

-- ==============================================================================
-- LOCATION SEARCH INDEXES
-- Full-text search on location names and descriptions
-- ==============================================================================

DEFINE INDEX location_name_ft ON location FIELDS name SEARCH ANALYZER narra_analyzer BM25;
DEFINE INDEX location_desc_ft ON location FIELDS description SEARCH ANALYZER narra_analyzer BM25;

-- ==============================================================================
-- EVENT SEARCH INDEXES
-- Full-text search on event titles and descriptions
-- ==============================================================================

DEFINE INDEX event_title_ft ON event FIELDS title SEARCH ANALYZER narra_analyzer BM25;
DEFINE INDEX event_desc_ft ON event FIELDS description SEARCH ANALYZER narra_analyzer BM25;

-- ==============================================================================
-- SCENE SEARCH INDEXES
-- Full-text search on scene titles and summaries
-- ==============================================================================

DEFINE INDEX scene_title_ft ON scene FIELDS title SEARCH ANALYZER narra_analyzer BM25;
DEFINE INDEX scene_summary_ft ON scene FIELDS summary SEARCH ANALYZER narra_analyzer BM25;

-- ==============================================================================
-- KNOWLEDGE SEARCH INDEXES
-- Full-text search on knowledge facts
-- ==============================================================================

DEFINE INDEX knowledge_fact_ft ON knowledge FIELDS fact SEARCH ANALYZER narra_analyzer BM25;
