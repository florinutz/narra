-- Arc Snapshots: Temporal embedding history for character arc tracking
-- Captures embedding vectors at each regeneration to track entity evolution.

DEFINE TABLE arc_snapshot SCHEMAFULL;

DEFINE FIELD entity_id ON arc_snapshot TYPE record;
DEFINE FIELD entity_type ON arc_snapshot TYPE string
    ASSERT $value IN ["character", "knowledge"];
DEFINE FIELD embedding ON arc_snapshot TYPE array<float>;
DEFINE FIELD delta_magnitude ON arc_snapshot TYPE option<float> DEFAULT NONE;
DEFINE FIELD event_id ON arc_snapshot TYPE option<record<event>> DEFAULT NONE;
DEFINE FIELD created_at ON arc_snapshot TYPE datetime VALUE time::now() READONLY;

DEFINE INDEX idx_arc_entity_time ON arc_snapshot FIELDS entity_id, created_at;
DEFINE INDEX idx_arc_entity_type ON arc_snapshot FIELDS entity_type;
