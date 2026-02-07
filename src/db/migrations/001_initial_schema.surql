-- Narra Initial Schema
-- Phase 1: Foundation tables for fiction worldbuilding

-- Characters
DEFINE TABLE character SCHEMAFULL;
DEFINE FIELD name ON character TYPE string;
DEFINE FIELD aliases ON character TYPE set<string> DEFAULT [];
DEFINE FIELD roles ON character TYPE set<string> DEFAULT [];
DEFINE FIELD created_at ON character TYPE datetime VALUE time::now() READONLY;
DEFINE FIELD updated_at ON character TYPE datetime VALUE time::now();
DEFINE INDEX idx_character_name ON character FIELDS name;

-- Locations
DEFINE TABLE location SCHEMAFULL;
DEFINE FIELD name ON location TYPE string;
DEFINE FIELD description ON location TYPE option<string>;
DEFINE FIELD loc_type ON location TYPE string;
DEFINE FIELD parent ON location TYPE option<record<location>>;
DEFINE FIELD created_at ON location TYPE datetime VALUE time::now() READONLY;
DEFINE FIELD updated_at ON location TYPE datetime VALUE time::now();
DEFINE INDEX idx_location_name ON location FIELDS name;
DEFINE INDEX idx_location_parent ON location FIELDS parent;

-- Timeline Events
DEFINE TABLE event SCHEMAFULL;
DEFINE FIELD title ON event TYPE string;
DEFINE FIELD description ON event TYPE option<string>;
DEFINE FIELD sequence ON event TYPE int;
DEFINE FIELD date ON event TYPE option<datetime>;
DEFINE FIELD date_precision ON event TYPE option<string>;
DEFINE FIELD duration_end ON event TYPE option<datetime>;
DEFINE FIELD created_at ON event TYPE datetime VALUE time::now() READONLY;
DEFINE FIELD updated_at ON event TYPE datetime VALUE time::now();
DEFINE INDEX idx_event_sequence ON event FIELDS sequence;
DEFINE INDEX idx_event_date ON event FIELDS date;

-- Character Relationships (graph edge)
DEFINE TABLE relates_to TYPE RELATION IN character OUT character SCHEMAFULL;
DEFINE FIELD rel_type ON relates_to TYPE string;
DEFINE FIELD subtype ON relates_to TYPE option<string>;
DEFINE FIELD label ON relates_to TYPE option<string>;
DEFINE FIELD created_at ON relates_to TYPE datetime VALUE time::now() READONLY;
DEFINE INDEX idx_relates_type ON relates_to FIELDS rel_type;

-- Knowledge States
DEFINE TABLE knowledge SCHEMAFULL;
DEFINE FIELD character ON knowledge TYPE record<character>;
DEFINE FIELD fact ON knowledge TYPE string;
DEFINE FIELD created_at ON knowledge TYPE datetime VALUE time::now() READONLY;
DEFINE INDEX idx_knowledge_character ON knowledge FIELDS character;
