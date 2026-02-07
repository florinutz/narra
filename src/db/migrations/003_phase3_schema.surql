-- Narra Phase 3 Schema Extension
-- Knowledge states with certainty and provenance
-- The knows edge connects character -> fact with rich metadata

-- ==============================================================================
-- KNOWS EDGE TABLE
-- Temporal knowledge: what a character knows about someone/something at a point in time
-- Supports tracking certainty levels, how knowledge was acquired, and provenance
-- ==============================================================================

DEFINE TABLE knows SCHEMAFULL TYPE RELATION IN character OUT character|knowledge;

-- Core knowledge fields
DEFINE FIELD certainty ON knows TYPE string;
DEFINE FIELD learning_method ON knows TYPE string;
DEFINE FIELD learned_at ON knows TYPE datetime DEFAULT time::now();

-- Provenance fields
DEFINE FIELD source_character ON knows TYPE option<record<character>>;
DEFINE FIELD event ON knows TYPE option<record<event>>;
DEFINE FIELD premises ON knows TYPE option<array<record<knows>>>;

-- For believes_wrongly: what the actual truth is
DEFINE FIELD truth_value ON knows TYPE option<string>;

-- Timestamps
DEFINE FIELD created_at ON knows TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON knows TYPE datetime DEFAULT time::now() VALUE time::now();

-- Index for temporal queries on character's knowledge
DEFINE INDEX idx_knows_character_time ON knows FIELDS in, learned_at;
-- Index for fact-based queries ("who knows about X?")
DEFINE INDEX idx_knows_out ON knows FIELDS out;
