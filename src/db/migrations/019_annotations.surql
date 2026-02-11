-- Annotation table: generic ML model outputs cached per entity.
-- One annotation per (entity_id, model_type) pair.

DEFINE TABLE IF NOT EXISTS annotation SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS entity_id ON annotation TYPE string;
DEFINE FIELD IF NOT EXISTS model_type ON annotation TYPE string;
DEFINE FIELD IF NOT EXISTS model_version ON annotation TYPE string;
DEFINE FIELD IF NOT EXISTS output ON annotation FLEXIBLE TYPE object;
DEFINE FIELD IF NOT EXISTS computed_at ON annotation TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS stale ON annotation TYPE bool DEFAULT false;
DEFINE INDEX IF NOT EXISTS idx_annotation_entity ON annotation FIELDS entity_id;
DEFINE INDEX IF NOT EXISTS idx_annotation_stale ON annotation FIELDS stale;
DEFINE INDEX IF NOT EXISTS idx_annotation_entity_model ON annotation FIELDS entity_id, model_type UNIQUE;
