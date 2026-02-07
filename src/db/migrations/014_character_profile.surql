-- Migrate character model from typed psychology fields to flexible profile

-- Remove old typed fields
REMOVE FIELD wounds ON character;
REMOVE FIELD desires ON character;
REMOVE FIELD contradictions ON character;

-- Add flexible profile field (allows arbitrary nested structure for fiction frameworks)
DEFINE FIELD profile ON character FLEXIBLE TYPE object DEFAULT {};
