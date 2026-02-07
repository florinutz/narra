-- Narra Phase 2 Schema Extension
-- Core Entities & Relationships: Character psychology, asymmetric relationships, scenes

-- ==============================================================================
-- CHARACTER PSYCHOLOGY FIELDS
-- Extending the character table with wound, desires, and contradictions
-- ==============================================================================

-- Wounds: Traumatic events that shape character behavior
-- Multiple wounds supported, with one marked as primary
DEFINE FIELD wounds ON character TYPE array<object> DEFAULT [];
DEFINE FIELD wounds.*.event ON character TYPE string;
DEFINE FIELD wounds.*.age ON character TYPE option<int>;
DEFINE FIELD wounds.*.belief ON character TYPE string;
DEFINE FIELD wounds.*.pattern ON character TYPE string;
DEFINE FIELD wounds.*.is_primary ON character TYPE bool DEFAULT false;

-- Desires: Conscious and unconscious goals that can change at timeline points
-- Uses event references for temporal versioning
DEFINE FIELD desires ON character TYPE array<object> DEFAULT [];
DEFINE FIELD desires.*.desire_type ON character TYPE string;
DEFINE FIELD desires.*.description ON character TYPE string;
DEFINE FIELD desires.*.valid_from ON character TYPE option<record<event>>;
DEFINE FIELD desires.*.valid_until ON character TYPE option<record<event>>;

-- Contradictions: Internal conflicts tagged by type
-- Types: value_vs_action, stated_vs_felt, public_vs_private (and custom)
DEFINE FIELD contradictions ON character TYPE array<object> DEFAULT [];
DEFINE FIELD contradictions.*.contradiction_type ON character TYPE string;
DEFINE FIELD contradictions.*.description ON character TYPE string;
DEFINE FIELD contradictions.*.is_primary ON character TYPE bool DEFAULT false;

-- ==============================================================================
-- PERCEIVES EDGE TABLE
-- Asymmetric relationship: A's view of B is independent of B's view of A
-- Each direction requires its own edge (RELATE A->perceives->B AND B->perceives->A)
-- ==============================================================================

DEFINE TABLE perceives TYPE RELATION IN character OUT character SCHEMAFULL;
DEFINE FIELD rel_types ON perceives TYPE set<string>;
DEFINE FIELD subtype ON perceives TYPE option<string>;
DEFINE FIELD feelings ON perceives TYPE option<string>;
DEFINE FIELD perception ON perceives TYPE option<string>;
DEFINE FIELD tension_level ON perceives TYPE option<int>;
DEFINE FIELD history_notes ON perceives TYPE option<string>;
DEFINE FIELD created_at ON perceives TYPE datetime VALUE time::now() READONLY;
DEFINE FIELD updated_at ON perceives TYPE datetime VALUE time::now();
DEFINE INDEX idx_perceives_types ON perceives FIELDS rel_types;

-- ==============================================================================
-- SCENE ENTITY
-- A narrative unit that happens at a specific event time and location
-- References timeline event for "when" and location for "where"
-- ==============================================================================

DEFINE TABLE scene SCHEMAFULL;
DEFINE FIELD title ON scene TYPE string;
DEFINE FIELD summary ON scene TYPE option<string>;
DEFINE FIELD event ON scene TYPE record<event>;
DEFINE FIELD primary_location ON scene TYPE record<location>;
DEFINE FIELD secondary_locations ON scene TYPE array<record<location>> DEFAULT [];
DEFINE FIELD created_at ON scene TYPE datetime VALUE time::now() READONLY;
DEFINE FIELD updated_at ON scene TYPE datetime VALUE time::now();
DEFINE INDEX idx_scene_event ON scene FIELDS event;
DEFINE INDEX idx_scene_location ON scene FIELDS primary_location;

-- ==============================================================================
-- PARTICIPATES_IN EDGE
-- Character participation in a scene with role
-- Roles: pov, protagonist, antagonist, supporting, witness, mentioned (or custom)
-- ==============================================================================

DEFINE TABLE participates_in TYPE RELATION IN character OUT scene SCHEMAFULL;
DEFINE FIELD role ON participates_in TYPE string;
DEFINE FIELD notes ON participates_in TYPE option<string>;
DEFINE FIELD created_at ON participates_in TYPE datetime VALUE time::now() READONLY;
DEFINE INDEX idx_participant_role ON participates_in FIELDS role;

-- ==============================================================================
-- INVOLVED_IN EDGE
-- Character involvement in timeline events
-- Tracks who participated in events and how they were affected
-- ==============================================================================

DEFINE TABLE involved_in TYPE RELATION IN character OUT event SCHEMAFULL;
DEFINE FIELD role ON involved_in TYPE option<string>;
DEFINE FIELD impact ON involved_in TYPE option<string>;
DEFINE FIELD created_at ON involved_in TYPE datetime VALUE time::now() READONLY;
