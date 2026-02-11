//! Composite text generation for entity embeddings.
//!
//! Generates natural-language descriptions of entities for embedding.
//! Composite text should be semantically rich but concise (50-200 words).

use crate::models::{Character, Event, Location, Note, Scene, UniverseFact};

/// Generate composite text for a character.
///
/// Creates a natural-language description combining character traits,
/// profile sections, relationships, and perceptions.
///
/// # Arguments
///
/// * `character` - The character entity
/// * `relationships` - Tuples of (relationship_type, target_name)
/// * `perceptions` - Tuples of (target_name, perception_text) from perceives edges
///
/// # Returns
///
/// A natural-language composite text suitable for embedding.
pub fn character_composite(
    character: &Character,
    relationships: &[(String, String)],
    perceptions: &[(String, String)],
) -> String {
    let mut parts = Vec::new();

    // Name and roles
    let roles_text = if !character.roles.is_empty() {
        format!("who is a {}", character.roles.join(", "))
    } else {
        "".to_string()
    };
    parts.push(format!("{} is a character {}", character.name, roles_text));

    // Aliases
    if !character.aliases.is_empty() {
        parts.push(format!("Also known as {}", character.aliases.join(", ")));
    }

    // Profile sections — iterate sorted keys for deterministic output
    let mut profile_keys: Vec<&String> = character.profile.keys().collect();
    profile_keys.sort();
    for key in profile_keys {
        if let Some(entries) = character.profile.get(key) {
            if !entries.is_empty() {
                let label = key.replace('_', " ");
                parts.push(format!("{}: {}", label, entries.join("; ")));
            }
        }
    }

    // Relationships
    if !relationships.is_empty() {
        let rel_desc: Vec<_> = relationships
            .iter()
            .map(|(rel_type, target)| format!("{} with {}", rel_type, target))
            .collect();
        parts.push(format!("They have relationships: {}", rel_desc.join(", ")));
    }

    // Perceptions (how this character sees others)
    if !perceptions.is_empty() {
        let perc_desc: Vec<_> = perceptions
            .iter()
            .map(|(target, text)| format!("sees {} as {}", target, text))
            .collect();
        parts.push(format!("They {}", perc_desc.join(", and ")));
    }

    // Join all parts with proper punctuation
    parts.join(". ") + "."
}

/// Generate composite text for a location.
///
/// # Arguments
///
/// * `location` - The location entity
///
/// # Returns
///
/// A natural-language composite text suitable for embedding.
pub fn location_composite(location: &Location) -> String {
    let description = location
        .description
        .as_ref()
        .map(|d| format!(". {}", d))
        .unwrap_or_default();

    format!(
        "{} is a {}{}.",
        location.name, location.loc_type, description
    )
}

/// Generate composite text for an event.
///
/// # Arguments
///
/// * `event` - The event entity
///
/// # Returns
///
/// A natural-language composite text suitable for embedding.
pub fn event_composite(event: &Event) -> String {
    let description = event
        .description
        .as_ref()
        .map(|d| format!(". {}", d))
        .unwrap_or_default();

    format!(
        "{}{}. Sequence: {}.",
        event.title, description, event.sequence
    )
}

/// Generate composite text for a scene.
///
/// # Arguments
///
/// * `scene` - The scene entity
/// * `event_title` - Optional title of the event this scene occurs during
/// * `location_name` - Optional name of the primary location
///
/// # Returns
///
/// A natural-language composite text suitable for embedding.
pub fn scene_composite(
    scene: &Scene,
    event_title: Option<&str>,
    location_name: Option<&str>,
) -> String {
    let mut parts = vec![scene.title.clone()];

    if let Some(summary) = &scene.summary {
        parts.push(summary.clone());
    }

    let mut context_parts = Vec::new();
    if let Some(location) = location_name {
        context_parts.push(format!("at {}", location));
    }
    if let Some(event) = event_title {
        context_parts.push(format!("during {}", event));
    }

    if !context_parts.is_empty() {
        parts.push(format!("Takes place {}", context_parts.join(" ")));
    }

    parts.join(". ") + "."
}

/// Generate composite text for a knowledge entity.
///
/// Certainty shapes the embedding semantically:
/// - "knows" → "{name} knows that {fact}"
/// - "believes_wrongly" → "{name} wrongly believes that {fact}"
/// - "suspects" → "{name} suspects that {fact}"
/// - "denies" → "{name} denies that {fact}"
/// - etc.
///
/// Learning method is appended for provenance context.
///
/// # Arguments
///
/// * `fact` - The knowledge fact text
/// * `character_name` - Name of the character who holds this knowledge
/// * `certainty` - Certainty level (knows, suspects, believes_wrongly, etc.)
/// * `learning_method` - How the knowledge was acquired (witnessed, told, etc.)
///
/// # Returns
///
/// A natural-language composite text suitable for embedding.
pub fn knowledge_composite(
    fact: &str,
    character_name: &str,
    certainty: &str,
    learning_method: Option<&str>,
) -> String {
    let certainty_phrase = match certainty {
        "knows" => format!("{} knows that {}", character_name, fact),
        "believes_wrongly" => format!("{} wrongly believes that {}", character_name, fact),
        "suspects" => format!("{} suspects that {}", character_name, fact),
        "denies" => format!("{} denies that {}", character_name, fact),
        "uncertain" => format!("{} is uncertain whether {}", character_name, fact),
        "assumes" => format!("{} assumes that {}", character_name, fact),
        "forgotten" => format!("{} has forgotten that {}", character_name, fact),
        _ => format!("{} knows that {}", character_name, fact),
    };

    let method_phrase = match learning_method {
        Some("witnessed") => " They witnessed this.",
        Some("told") => " They were told this.",
        Some("overheard") => " They overheard this.",
        Some("discovered") => " They discovered this.",
        Some("deduced") => " They deduced this.",
        Some("read") => " They read about this.",
        Some("remembered") => " They remembered this.",
        Some("initial") => " They knew this from the start.",
        _ => "",
    };

    format!("{}.{}", certainty_phrase, method_phrase)
}

/// Generate composite text for a perspective (observer's view of a target).
///
/// Combines perception data, observer's knowledge about the target,
/// and shared scene experiences into a rich semantic description.
///
/// # Arguments
///
/// * `observer_name` - Name of the observing character
/// * `target_name` - Name of the observed character
/// * `rel_types` - Relationship types between observer and target
/// * `subtype` - Optional relationship subtype
/// * `feelings` - How the observer feels about the target
/// * `perception` - How the observer perceives the target
/// * `tension_level` - Tension level (0-10)
/// * `history_notes` - Historical notes about their relationship
/// * `knowledge` - Observer's knowledge about target: (fact, certainty)
/// * `shared_scenes` - Scenes they shared: (title, summary)
#[allow(clippy::too_many_arguments)]
pub fn perspective_composite(
    observer_name: &str,
    target_name: &str,
    rel_types: &[String],
    subtype: Option<&str>,
    feelings: Option<&str>,
    perception: Option<&str>,
    tension_level: Option<i32>,
    history_notes: Option<&str>,
    knowledge: &[(String, String)],
    shared_scenes: &[(String, Option<String>)],
) -> String {
    let mut parts = Vec::new();

    // Core identity
    parts.push(format!(
        "{}'s perspective on {}",
        observer_name, target_name
    ));

    // Relationship types
    if !rel_types.is_empty() {
        let rel_str = rel_types.join(", ");
        if let Some(st) = subtype {
            parts.push(format!(
                "{} has a {} ({}) relationship with {}",
                observer_name, rel_str, st, target_name
            ));
        } else {
            parts.push(format!(
                "{} has a {} relationship with {}",
                observer_name, rel_str, target_name
            ));
        }
    }

    // Feelings
    if let Some(f) = feelings {
        parts.push(format!("{} feels that {}", observer_name, f));
    }

    // Perception
    if let Some(p) = perception {
        parts.push(format!(
            "{} perceives {} as {}",
            observer_name, target_name, p
        ));
    }

    // Tension level
    if let Some(t) = tension_level {
        parts.push(format!("Tension level: {}/10", t));
    }

    // History notes
    if let Some(h) = history_notes {
        parts.push(format!("History: {}", h));
    }

    // Knowledge about target (certainty-aware phrasing)
    for (fact, certainty) in knowledge {
        let phrase = match certainty.as_str() {
            "knows" => format!("{} knows that {}", observer_name, fact),
            "believes_wrongly" => format!("{} wrongly believes that {}", observer_name, fact),
            "suspects" => format!("{} suspects that {}", observer_name, fact),
            "denies" => format!("{} denies that {}", observer_name, fact),
            "uncertain" => format!("{} is uncertain whether {}", observer_name, fact),
            "assumes" => format!("{} assumes that {}", observer_name, fact),
            "forgotten" => format!("{} has forgotten that {}", observer_name, fact),
            _ => format!("{} knows that {}", observer_name, fact),
        };
        parts.push(phrase);
    }

    // Shared scenes
    if !shared_scenes.is_empty() {
        let scene_descs: Vec<String> = shared_scenes
            .iter()
            .map(|(title, summary)| {
                if let Some(s) = summary {
                    format!("\"{}\" - {}", title, s)
                } else {
                    format!("\"{}\"", title)
                }
            })
            .collect();
        parts.push(format!(
            "They shared experiences: {}",
            scene_descs.join("; ")
        ));
    }

    parts.join(". ") + "."
}

/// Generate composite text for a relationship (relates_to edge).
///
/// Creates a natural-language description of the relationship between
/// two characters, including their roles and any labels.
///
/// # Arguments
///
/// * `from_name` - Name of the source character
/// * `from_roles` - Roles of the source character
/// * `to_name` - Name of the target character
/// * `to_roles` - Roles of the target character
/// * `rel_type` - Relationship type (e.g., "family", "rival")
/// * `subtype` - Optional subtype (e.g., "sibling")
/// * `label` - Optional freeform label describing the relationship
pub fn relationship_composite(
    from_name: &str,
    from_roles: &[String],
    to_name: &str,
    to_roles: &[String],
    rel_type: &str,
    subtype: Option<&str>,
    label: Option<&str>,
) -> String {
    let mut parts = Vec::new();

    // Core relationship identity
    let from_roles_str = if !from_roles.is_empty() {
        format!(" ({})", from_roles.join(", "))
    } else {
        String::new()
    };

    let to_roles_str = if !to_roles.is_empty() {
        format!(" ({})", to_roles.join(", "))
    } else {
        String::new()
    };

    parts.push(format!(
        "Relationship between {}{} and {}{}: {} bond",
        from_name, from_roles_str, to_name, to_roles_str, rel_type
    ));

    // Subtype
    if let Some(st) = subtype {
        parts.push(format!("Subtype: {}", st));
    }

    // Label (freeform description)
    if let Some(lbl) = label {
        parts.push(lbl.to_string());
    }

    parts.join(". ") + "."
}

/// Generate identity facet composite for a character.
///
/// Covers: name, roles, aliases (~15-30 words).
/// Provides stable, core identity that rarely changes.
///
/// # Arguments
///
/// * `character` - The character entity
///
/// # Returns
///
/// A natural-language composite text for the identity facet.
pub fn identity_composite(character: &Character) -> String {
    let mut parts = Vec::new();

    // Name and roles
    if !character.roles.is_empty() {
        parts.push(format!(
            "{} is a {}",
            character.name,
            character.roles.join(", ")
        ));
    } else {
        parts.push(format!("{} is a character", character.name));
    }

    // Aliases
    if !character.aliases.is_empty() {
        parts.push(format!("Also known as {}", character.aliases.join(", ")));
    }

    parts.join(". ") + "."
}

/// Generate psychology facet composite for a character.
///
/// Covers: profile categories only (wound, desire_conscious, desire_unconscious,
/// contradiction, secret, etc.).
/// Provides psychological depth without social or narrative context.
///
/// # Arguments
///
/// * `character` - The character entity
///
/// # Returns
///
/// A natural-language composite text for the psychology facet.
pub fn psychology_composite(character: &Character) -> String {
    if character.profile.is_empty() {
        return format!("{} has no defined psychological profile.", character.name);
    }

    let mut parts = Vec::new();
    parts.push(format!("{}'s psychology", character.name));

    // Profile sections — iterate sorted keys for deterministic output
    let mut profile_keys: Vec<&String> = character.profile.keys().collect();
    profile_keys.sort();
    for key in profile_keys {
        if let Some(entries) = character.profile.get(key) {
            if !entries.is_empty() {
                let label = key.replace('_', " ");
                parts.push(format!("{}: {}", label, entries.join("; ")));
            }
        }
    }

    parts.join(". ") + "."
}

/// Generate social facet composite for a character.
///
/// Covers: relationships + how others perceive this character (inbound perceptions).
/// Provides social network and reputation context.
///
/// # Arguments
///
/// * `character` - The character entity
/// * `relationships` - Tuples of (relationship_type, target_name)
/// * `perceptions_of` - Inbound perceptions: (observer_name, perception_text)
///
/// # Returns
///
/// A natural-language composite text for the social facet.
pub fn social_composite(
    character: &Character,
    relationships: &[(String, String)],
    perceptions_of: &[(String, String)],
) -> String {
    let mut parts = Vec::new();
    parts.push(format!("{}'s social network", character.name));

    // Relationships
    if !relationships.is_empty() {
        let rel_desc: Vec<_> = relationships
            .iter()
            .map(|(rel_type, target)| format!("{} with {}", rel_type, target))
            .collect();
        parts.push(format!("Relationships: {}", rel_desc.join(", ")));
    }

    // Inbound perceptions (how others see this character)
    if !perceptions_of.is_empty() {
        let perc_desc: Vec<_> = perceptions_of
            .iter()
            .map(|(observer, text)| format!("{} sees them as {}", observer, text))
            .collect();
        parts.push(format!("Perceived by others: {}", perc_desc.join("; ")));
    }

    if relationships.is_empty() && perceptions_of.is_empty() {
        parts.push("No relationships or perceptions recorded".to_string());
    }

    parts.join(". ") + "."
}

/// Generate narrative facet composite for a character.
///
/// Covers: scenes + knowledge held by character.
/// Provides narrative involvement and what the character knows.
///
/// # Arguments
///
/// * `character_name` - Name of the character
/// * `scenes` - Tuples of (scene_title, optional summary)
/// * `knowledge` - Tuples of (fact, certainty)
///
/// # Returns
///
/// A natural-language composite text for the narrative facet.
pub fn narrative_composite(
    character_name: &str,
    scenes: &[(String, Option<String>)],
    knowledge: &[(String, String)],
) -> String {
    let mut parts = Vec::new();
    parts.push(format!("{}'s narrative involvement", character_name));

    // Scene participation
    if !scenes.is_empty() {
        let scene_descs: Vec<String> = scenes
            .iter()
            .map(|(title, summary)| {
                if let Some(s) = summary {
                    format!("\"{}\" - {}", title, s)
                } else {
                    format!("\"{}\"", title)
                }
            })
            .collect();
        parts.push(format!("Participates in: {}", scene_descs.join("; ")));
    }

    // Knowledge held
    if !knowledge.is_empty() {
        let knowledge_descs: Vec<String> = knowledge
            .iter()
            .map(|(fact, certainty)| match certainty.as_str() {
                "knows" => format!("knows that {}", fact),
                "believes_wrongly" => format!("wrongly believes that {}", fact),
                "suspects" => format!("suspects that {}", fact),
                "denies" => format!("denies that {}", fact),
                "uncertain" => format!("is uncertain whether {}", fact),
                "assumes" => format!("assumes that {}", fact),
                "forgotten" => format!("has forgotten that {}", fact),
                _ => format!("knows that {}", fact),
            })
            .collect();
        parts.push(format!("{} {}", character_name, knowledge_descs.join("; ")));
    }

    if scenes.is_empty() && knowledge.is_empty() {
        parts.push("No scenes or knowledge recorded".to_string());
    }

    parts.join(". ") + "."
}

/// Generate composite text for a note.
///
/// Combines title and body, truncating body to ~200 words for embedding.
///
/// # Arguments
///
/// * `note` - The note entity
///
/// # Returns
///
/// A natural-language composite text suitable for embedding.
pub fn note_composite(note: &Note) -> String {
    let truncated_body = truncate_words(&note.body, 200);
    format!("{}: {}", note.title, truncated_body)
}

/// Generate composite text for a universe fact.
///
/// Combines title, category, description, and enforcement level.
///
/// # Arguments
///
/// * `fact` - The universe fact entity
///
/// # Returns
///
/// A natural-language composite text suitable for embedding.
pub fn fact_composite(fact: &UniverseFact) -> String {
    use crate::models::fact::{EnforcementLevel, FactCategory};

    let mut parts = Vec::new();

    // Title with categories
    let categories: Vec<&str> = fact
        .categories
        .iter()
        .map(|c| match c {
            FactCategory::PhysicsMagic => "physics/magic",
            FactCategory::SocialCultural => "social/cultural",
            FactCategory::Technology => "technology",
            FactCategory::Custom(s) => s.as_str(),
        })
        .collect();
    if !categories.is_empty() {
        parts.push(format!("{} ({})", fact.title, categories.join(", ")));
    } else {
        parts.push(fact.title.clone());
    }

    // Description
    parts.push(fact.description.clone());

    // Enforcement level
    let enforcement = match fact.enforcement_level {
        EnforcementLevel::Informational => "informational",
        EnforcementLevel::Warning => "warning",
        EnforcementLevel::Strict => "strict",
    };
    parts.push(format!("Enforcement: {}", enforcement));

    parts.join(". ") + "."
}

/// Truncate a string to approximately `max_words` words.
fn truncate_words(text: &str, max_words: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= max_words {
        text.to_string()
    } else {
        words[..max_words].join(" ") + "..."
    }
}

/// Generate composite text from raw JSON entity data.
///
/// This is a convenience function for cases where you have raw JSON
/// (e.g., during backfill operations).
///
/// # Arguments
///
/// * `entity_type` - The entity type ("character", "location", "event", "scene")
/// * `entity_json` - The entity data as JSON
///
/// # Returns
///
/// A natural-language composite text suitable for embedding.
pub fn generate_composite_text(entity_type: &str, entity_json: &serde_json::Value) -> String {
    match entity_type {
        "character" => {
            let name = entity_json["name"].as_str().unwrap_or("Unknown");
            let roles = entity_json["roles"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(String::from)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let roles_text = if !roles.is_empty() {
                format!("who is a {}", roles.join(", "))
            } else {
                "".to_string()
            };

            format!("{} is a character {}.", name, roles_text)
        }
        "location" => {
            let name = entity_json["name"].as_str().unwrap_or("Unknown");
            let loc_type = entity_json["loc_type"].as_str().unwrap_or("place");
            let description = entity_json["description"]
                .as_str()
                .map(|d| format!(". {}", d))
                .unwrap_or_default();

            format!("{} is a {}{}.", name, loc_type, description)
        }
        "event" => {
            let title = entity_json["title"].as_str().unwrap_or("Unknown");
            let description = entity_json["description"]
                .as_str()
                .map(|d| format!(". {}", d))
                .unwrap_or_default();
            let sequence = entity_json["sequence"].as_i64().unwrap_or(0);

            format!("{}{}. Sequence: {}.", title, description, sequence)
        }
        "scene" => {
            let title = entity_json["title"].as_str().unwrap_or("Unknown");
            let summary = entity_json["summary"]
                .as_str()
                .map(|s| format!(". {}", s))
                .unwrap_or_default();

            format!("{}{}.", title, summary)
        }
        "knowledge" => {
            let fact = entity_json["fact"].as_str().unwrap_or("Unknown");
            let character_name = entity_json["character_name"].as_str().unwrap_or("Someone");
            let certainty = entity_json["certainty"].as_str().unwrap_or("knows");
            let learning_method = entity_json["learning_method"].as_str();

            knowledge_composite(fact, character_name, certainty, learning_method)
        }
        "note" => {
            let title = entity_json["title"].as_str().unwrap_or("Unknown");
            let body = entity_json["body"].as_str().unwrap_or("");
            let truncated_body = truncate_words(body, 200);
            format!("{}: {}", title, truncated_body)
        }
        "fact" => {
            let title = entity_json["title"].as_str().unwrap_or("Unknown");
            let description = entity_json["description"].as_str().unwrap_or("");
            let categories = entity_json["categories"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            let enforcement = entity_json["enforcement_level"]
                .as_str()
                .unwrap_or("warning");

            if categories.is_empty() {
                format!("{}. {}. Enforcement: {}.", title, description, enforcement)
            } else {
                format!(
                    "{} ({}). {}. Enforcement: {}.",
                    title, categories, description, enforcement
                )
            }
        }
        _ => format!("Unknown entity type: {}", entity_type),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use surrealdb::RecordId;

    #[test]
    fn test_character_composite_minimal() {
        let character = Character {
            id: RecordId::from(("character", "test")),
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec!["warrior".to_string()],
            profile: HashMap::new(),
            created_at: surrealdb::Datetime::default(),
            updated_at: surrealdb::Datetime::default(),
        };

        let result = character_composite(&character, &[], &[]);
        assert!(result.contains("Alice"));
        assert!(result.contains("warrior"));
    }

    #[test]
    fn test_character_composite_full() {
        let mut profile = HashMap::new();
        profile.insert(
            "wound".to_string(),
            vec!["I am unlovable — pushes people away".to_string()],
        );
        profile.insert("desire_conscious".to_string(), vec!["freedom".to_string()]);
        profile.insert(
            "desire_unconscious".to_string(),
            vec!["belonging".to_string()],
        );
        profile.insert(
            "contradiction".to_string(),
            vec!["seeks connection while pushing away".to_string()],
        );

        let character = Character {
            id: RecordId::from(("character", "test")),
            name: "Alice".to_string(),
            aliases: vec!["The Blade".to_string()],
            roles: vec!["warrior".to_string(), "leader".to_string()],
            profile,
            created_at: surrealdb::Datetime::default(),
            updated_at: surrealdb::Datetime::default(),
        };

        let relationships = vec![
            ("allied".to_string(), "Bob".to_string()),
            ("rival".to_string(), "Carol".to_string()),
        ];

        let perceptions = vec![("Bob".to_string(), "a trusted ally but naive".to_string())];

        let result = character_composite(&character, &relationships, &perceptions);
        assert!(result.contains("Alice"));
        assert!(result.contains("The Blade"));
        assert!(result.contains("warrior"));
        assert!(result.contains("leader"));
        assert!(result.contains("unlovable"));
        assert!(result.contains("freedom"));
        assert!(result.contains("belonging"));
        assert!(result.contains("allied with Bob"));
        assert!(result.contains("sees Bob as a trusted ally"));
    }

    #[test]
    fn test_perspective_composite_full() {
        let knowledge = vec![
            (
                "Alice is secretly the heir".to_string(),
                "suspects".to_string(),
            ),
            (
                "Alice trained under the old master".to_string(),
                "knows".to_string(),
            ),
        ];
        let shared_scenes = vec![
            (
                "The Confrontation".to_string(),
                Some("Alice and Bob face off".to_string()),
            ),
            ("The Tavern Meeting".to_string(), None),
        ];

        let result = perspective_composite(
            "Bob",
            "Alice",
            &["ally".to_string(), "rival".to_string()],
            Some("frenemy"),
            Some("conflicted about her true motives"),
            Some("dangerous but trustworthy"),
            Some(7),
            Some("They grew up together in the village"),
            &knowledge,
            &shared_scenes,
        );

        assert!(result.contains("Bob's perspective on Alice"));
        assert!(result.contains("ally, rival"));
        assert!(result.contains("frenemy"));
        assert!(result.contains("conflicted about her true motives"));
        assert!(result.contains("dangerous but trustworthy"));
        assert!(result.contains("7/10"));
        assert!(result.contains("grew up together"));
        assert!(result.contains("Bob suspects that Alice is secretly the heir"));
        assert!(result.contains("Bob knows that Alice trained under the old master"));
        assert!(result.contains("The Confrontation"));
        assert!(result.contains("The Tavern Meeting"));
    }

    #[test]
    fn test_perspective_composite_minimal() {
        let result =
            perspective_composite("Bob", "Alice", &[], None, None, None, None, None, &[], &[]);

        assert!(result.contains("Bob's perspective on Alice"));
        // Should still form valid text
        assert!(result.ends_with('.'));
    }

    #[test]
    fn test_relationship_composite_full() {
        let result = relationship_composite(
            "Alice",
            &["warrior".to_string(), "protagonist".to_string()],
            "Bob",
            &["sage".to_string(), "mentor".to_string()],
            "family",
            Some("sibling"),
            Some("Alice's younger brother who she protects fiercely"),
        );
        assert!(result.contains(
            "Relationship between Alice (warrior, protagonist) and Bob (sage, mentor): family bond"
        ));
        assert!(result.contains("Subtype: sibling"));
        assert!(result.contains("Alice's younger brother"));
    }

    #[test]
    fn test_relationship_composite_minimal() {
        let result = relationship_composite("Alice", &[], "Bob", &[], "rival", None, None);
        assert!(result.contains("Relationship between Alice and Bob: rival bond"));
        assert!(result.ends_with('.'));
    }

    #[test]
    fn test_location_composite() {
        let location = Location {
            id: RecordId::from(("location", "test")),
            name: "The Dark Forest".to_string(),
            description: Some("A dense, ancient forest shrouded in perpetual twilight".to_string()),
            loc_type: "place".to_string(),
            parent: None,
            created_at: surrealdb::Datetime::default(),
            updated_at: surrealdb::Datetime::default(),
        };

        let result = location_composite(&location);
        assert!(result.contains("The Dark Forest"));
        assert!(result.contains("dense"));
    }

    #[test]
    fn test_event_composite() {
        let event = Event {
            id: RecordId::from(("event", "test")),
            title: "The Great Betrayal".to_string(),
            description: Some("Bob reveals his true allegiance".to_string()),
            sequence: 5,
            date: None,
            date_precision: None,
            duration_end: None,
            created_at: surrealdb::Datetime::default(),
            updated_at: surrealdb::Datetime::default(),
        };

        let result = event_composite(&event);
        assert!(result.contains("The Great Betrayal"));
        assert!(result.contains("allegiance"));
        assert!(result.contains("Sequence: 5"));
    }

    #[test]
    fn test_scene_composite() {
        let scene = Scene {
            id: RecordId::from(("scene", "test")),
            title: "The Confrontation".to_string(),
            summary: Some("Alice confronts Bob about his betrayal".to_string()),
            event: RecordId::from(("event", "betrayal")),
            primary_location: RecordId::from(("location", "forest")),
            secondary_locations: vec![],
            created_at: surrealdb::Datetime::default(),
            updated_at: surrealdb::Datetime::default(),
        };

        let result = scene_composite(&scene, Some("The Great Betrayal"), Some("The Dark Forest"));
        assert!(result.contains("The Confrontation"));
        assert!(result.contains("The Dark Forest"));
        assert!(result.contains("The Great Betrayal"));
    }

    #[test]
    fn test_identity_composite_minimal() {
        let character = Character {
            id: RecordId::from(("character", "test")),
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec![],
            profile: HashMap::new(),
            created_at: surrealdb::Datetime::default(),
            updated_at: surrealdb::Datetime::default(),
        };

        let result = identity_composite(&character);
        assert!(result.contains("Alice"));
        assert!(result.contains("is a character"));
    }

    #[test]
    fn test_identity_composite_full() {
        let character = Character {
            id: RecordId::from(("character", "test")),
            name: "Alice".to_string(),
            aliases: vec!["The Blade".to_string(), "Shadow".to_string()],
            roles: vec!["warrior".to_string(), "leader".to_string()],
            profile: HashMap::new(),
            created_at: surrealdb::Datetime::default(),
            updated_at: surrealdb::Datetime::default(),
        };

        let result = identity_composite(&character);
        assert!(result.contains("Alice"));
        assert!(result.contains("warrior"));
        assert!(result.contains("leader"));
        assert!(result.contains("The Blade"));
        assert!(result.contains("Shadow"));
    }

    #[test]
    fn test_psychology_composite_empty() {
        let character = Character {
            id: RecordId::from(("character", "test")),
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec![],
            profile: HashMap::new(),
            created_at: surrealdb::Datetime::default(),
            updated_at: surrealdb::Datetime::default(),
        };

        let result = psychology_composite(&character);
        assert!(result.contains("Alice"));
        assert!(result.contains("no defined psychological profile"));
    }

    #[test]
    fn test_psychology_composite_full() {
        let mut profile = HashMap::new();
        profile.insert(
            "wound".to_string(),
            vec!["I am unlovable — pushes people away".to_string()],
        );
        profile.insert("desire_conscious".to_string(), vec!["freedom".to_string()]);
        profile.insert(
            "desire_unconscious".to_string(),
            vec!["belonging".to_string()],
        );
        profile.insert(
            "contradiction".to_string(),
            vec!["seeks connection while pushing away".to_string()],
        );

        let character = Character {
            id: RecordId::from(("character", "test")),
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec![],
            profile,
            created_at: surrealdb::Datetime::default(),
            updated_at: surrealdb::Datetime::default(),
        };

        let result = psychology_composite(&character);
        assert!(result.contains("Alice's psychology"));
        assert!(result.contains("unlovable"));
        assert!(result.contains("freedom"));
        assert!(result.contains("belonging"));
        assert!(result.contains("contradiction"));
    }

    #[test]
    fn test_social_composite_empty() {
        let character = Character {
            id: RecordId::from(("character", "test")),
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec![],
            profile: HashMap::new(),
            created_at: surrealdb::Datetime::default(),
            updated_at: surrealdb::Datetime::default(),
        };

        let result = social_composite(&character, &[], &[]);
        assert!(result.contains("Alice's social network"));
        assert!(result.contains("No relationships or perceptions"));
    }

    #[test]
    fn test_social_composite_full() {
        let character = Character {
            id: RecordId::from(("character", "test")),
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec![],
            profile: HashMap::new(),
            created_at: surrealdb::Datetime::default(),
            updated_at: surrealdb::Datetime::default(),
        };

        let relationships = vec![
            ("allied".to_string(), "Bob".to_string()),
            ("rival".to_string(), "Carol".to_string()),
        ];

        let perceptions_of = vec![
            ("Bob".to_string(), "a trusted leader".to_string()),
            (
                "Carol".to_string(),
                "dangerous and unpredictable".to_string(),
            ),
        ];

        let result = social_composite(&character, &relationships, &perceptions_of);
        assert!(result.contains("Alice's social network"));
        assert!(result.contains("allied with Bob"));
        assert!(result.contains("rival with Carol"));
        assert!(result.contains("Bob sees them as a trusted leader"));
        assert!(result.contains("Carol sees them as dangerous"));
    }

    #[test]
    fn test_narrative_composite_empty() {
        let result = narrative_composite("Alice", &[], &[]);
        assert!(result.contains("Alice's narrative involvement"));
        assert!(result.contains("No scenes or knowledge"));
    }

    #[test]
    fn test_narrative_composite_full() {
        let scenes = vec![
            (
                "The Confrontation".to_string(),
                Some("Alice faces Bob".to_string()),
            ),
            ("The Tavern Meeting".to_string(), None),
        ];

        let knowledge = vec![
            (
                "Bob is secretly the heir".to_string(),
                "suspects".to_string(),
            ),
            (
                "Carol trained under the old master".to_string(),
                "knows".to_string(),
            ),
        ];

        let result = narrative_composite("Alice", &scenes, &knowledge);
        assert!(result.contains("Alice's narrative involvement"));
        assert!(result.contains("The Confrontation"));
        assert!(result.contains("The Tavern Meeting"));
        assert!(result.contains("suspects that Bob is secretly the heir"));
        assert!(result.contains("knows that Carol trained under the old master"));
    }
}
