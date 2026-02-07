-- Phase 11: Universe Facts for world rules and constraints
-- Facts are first-class entities that define world rules with flexible enforcement

DEFINE TABLE universe_fact SCHEMAFULL;
DEFINE FIELD title ON universe_fact TYPE string;
DEFINE FIELD description ON universe_fact TYPE string;
DEFINE FIELD categories ON universe_fact TYPE array;
DEFINE FIELD categories.* ON universe_fact TYPE string;
DEFINE FIELD enforcement_level ON universe_fact TYPE string;
DEFINE FIELD scope ON universe_fact FLEXIBLE;
DEFINE FIELD created_at ON universe_fact TYPE datetime VALUE time::now() READONLY;
DEFINE FIELD updated_at ON universe_fact TYPE datetime VALUE time::now();
DEFINE INDEX idx_fact_title ON universe_fact FIELDS title;

-- FULLTEXT search indexes for facts (same pattern as notes)
DEFINE ANALYZER fact_analyzer TOKENIZERS class FILTERS ascii, lowercase, snowball(english);
DEFINE INDEX idx_fact_search_title ON universe_fact FIELDS title SEARCH ANALYZER fact_analyzer BM25;
DEFINE INDEX idx_fact_search_description ON universe_fact FIELDS description SEARCH ANALYZER fact_analyzer BM25;

-- Fact application edge (fact -> any entity)
-- Uses union type to allow linking to all entity types
DEFINE TABLE applies_to TYPE RELATION IN universe_fact OUT character|location|event|scene|knowledge SCHEMAFULL;
DEFINE FIELD link_type ON applies_to TYPE string;
DEFINE FIELD confidence ON applies_to TYPE option<float>;
DEFINE FIELD created_at ON applies_to TYPE datetime VALUE time::now() READONLY;
