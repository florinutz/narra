-- Phase table: persisted narrative phase detection results
DEFINE TABLE IF NOT EXISTS phase SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS name ON phase TYPE string;
DEFINE FIELD IF NOT EXISTS label ON phase TYPE string;
DEFINE FIELD IF NOT EXISTS phase_order ON phase TYPE int;
DEFINE FIELD IF NOT EXISTS sequence_range_min ON phase TYPE option<int> DEFAULT NONE;
DEFINE FIELD IF NOT EXISTS sequence_range_max ON phase TYPE option<int> DEFAULT NONE;
DEFINE FIELD IF NOT EXISTS entity_type_counts ON phase TYPE object DEFAULT {};
DEFINE FIELD IF NOT EXISTS weights_content ON phase TYPE float DEFAULT 0.6;
DEFINE FIELD IF NOT EXISTS weights_neighborhood ON phase TYPE float DEFAULT 0.25;
DEFINE FIELD IF NOT EXISTS weights_temporal ON phase TYPE float DEFAULT 0.15;
DEFINE FIELD IF NOT EXISTS member_count ON phase TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS created_at ON phase TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON phase TYPE datetime DEFAULT time::now() VALUE time::now();
DEFINE INDEX IF NOT EXISTS idx_phase_order ON phase FIELDS phase_order;

-- Membership edge: entity belongs to a phase (denormalized for fast loading)
DEFINE TABLE IF NOT EXISTS belongs_to_phase TYPE RELATION SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS in ON belongs_to_phase TYPE record;
DEFINE FIELD IF NOT EXISTS out ON belongs_to_phase TYPE record<phase>;
DEFINE FIELD IF NOT EXISTS entity_type ON belongs_to_phase TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS entity_name ON belongs_to_phase TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS centrality ON belongs_to_phase TYPE float DEFAULT 0.0;
DEFINE FIELD IF NOT EXISTS sequence_position ON belongs_to_phase TYPE option<float> DEFAULT NONE;
DEFINE INDEX IF NOT EXISTS idx_btp_phase ON belongs_to_phase FIELDS out;
DEFINE INDEX IF NOT EXISTS idx_btp_entity ON belongs_to_phase FIELDS in;
