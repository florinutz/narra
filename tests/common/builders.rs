//! Test data builders for entity construction.
//!
//! Provides fluent API for creating test entities with sensible defaults.

use narra::models::{CharacterCreate, EventCreate, KnowledgeCreate, LocationCreate, SceneCreate};
use surrealdb::RecordId;

/// Builder for creating test characters.
pub struct CharacterBuilder {
    name: String,
    aliases: Vec<String>,
    roles: Vec<String>,
}

impl CharacterBuilder {
    /// Create a new character builder with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            aliases: Vec::new(),
            roles: vec!["character".to_string()],
        }
    }

    /// Add an alias for the character.
    pub fn alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }

    /// Add a role for the character.
    pub fn role(mut self, role: impl Into<String>) -> Self {
        self.roles.push(role.into());
        self
    }

    /// Build the CharacterCreate struct.
    pub fn build(self) -> CharacterCreate {
        CharacterCreate {
            name: self.name,
            aliases: self.aliases,
            roles: self.roles,
            ..Default::default()
        }
    }
}

/// Builder for creating test locations.
pub struct LocationBuilder {
    name: String,
    description: Option<String>,
    loc_type: String,
    parent: Option<RecordId>,
}

impl LocationBuilder {
    /// Create a new location builder with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            loc_type: "place".to_string(),
            parent: None,
        }
    }

    /// Set the location description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the location type (e.g., "city", "building", "room").
    pub fn loc_type(mut self, loc_type: impl Into<String>) -> Self {
        self.loc_type = loc_type.into();
        self
    }

    /// Set the parent location.
    pub fn parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent = Some(RecordId::from(("location", parent_id.into().as_str())));
        self
    }

    /// Build the LocationCreate struct.
    pub fn build(self) -> LocationCreate {
        LocationCreate {
            name: self.name,
            description: self.description,
            loc_type: self.loc_type,
            parent: self.parent,
        }
    }
}

/// Builder for creating test events.
pub struct EventBuilder {
    title: String,
    description: Option<String>,
    sequence: i64,
}

impl EventBuilder {
    /// Create a new event builder with the given title.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            description: None,
            sequence: 0,
        }
    }

    /// Set the event description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the sequence number for timeline ordering.
    pub fn sequence(mut self, seq: i64) -> Self {
        self.sequence = seq;
        self
    }

    /// Build the EventCreate struct.
    pub fn build(self) -> EventCreate {
        EventCreate {
            title: self.title,
            description: self.description,
            sequence: self.sequence,
            date: None,
            date_precision: None,
            duration_end: None,
        }
    }
}

/// Builder for creating test scenes.
///
/// Scenes require an event and primary location, provided via their IDs.
pub struct SceneBuilder {
    title: String,
    event_id: String,
    primary_location_id: String,
    summary: Option<String>,
    secondary_locations: Vec<RecordId>,
}

impl SceneBuilder {
    /// Create a new scene builder with required fields.
    ///
    /// # Arguments
    ///
    /// * `title` - Scene title
    /// * `event_id` - Event ID (key part only, e.g., "abc123")
    /// * `location_id` - Primary location ID (key part only, e.g., "xyz789")
    pub fn new(
        title: impl Into<String>,
        event_id: impl Into<String>,
        location_id: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            event_id: event_id.into(),
            primary_location_id: location_id.into(),
            summary: None,
            secondary_locations: Vec::new(),
        }
    }

    /// Set the scene summary.
    pub fn summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Add a secondary location.
    pub fn secondary_location(mut self, location_id: impl Into<String>) -> Self {
        self.secondary_locations
            .push(RecordId::from(("location", location_id.into().as_str())));
        self
    }

    /// Build the SceneCreate struct.
    pub fn build(self) -> SceneCreate {
        SceneCreate {
            title: self.title,
            summary: self.summary,
            event: RecordId::from(("event", self.event_id.as_str())),
            primary_location: RecordId::from(("location", self.primary_location_id.as_str())),
            secondary_locations: self.secondary_locations,
        }
    }
}

/// Builder for creating test knowledge facts.
///
/// Knowledge facts are associated with a character. For integration tests,
/// typically create a "narrator" or system character to own standalone facts.
pub struct KnowledgeBuilder {
    fact: String,
    character_id: String,
}

impl KnowledgeBuilder {
    /// Create a new knowledge builder with the given fact.
    pub fn new(fact: impl Into<String>) -> Self {
        Self {
            fact: fact.into(),
            character_id: "narrator".to_string(),
        }
    }

    /// Set the character who "owns" this knowledge fact.
    pub fn for_character(mut self, character_id: impl Into<String>) -> Self {
        self.character_id = character_id.into();
        self
    }

    /// Build the KnowledgeCreate struct.
    pub fn build(self) -> KnowledgeCreate {
        KnowledgeCreate {
            character: RecordId::from(("character", self.character_id.as_str())),
            fact: self.fact,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_character_builder() {
        let data = CharacterBuilder::new("Alice")
            .alias("The Shadow")
            .role("protagonist")
            .build();

        assert_eq!(data.name, "Alice");
        assert!(data.aliases.contains(&"The Shadow".to_string()));
        assert!(data.roles.contains(&"protagonist".to_string()));
    }

    #[test]
    fn test_location_builder() {
        let data = LocationBuilder::new("The Tower")
            .description("A tall stone tower")
            .loc_type("building")
            .build();

        assert_eq!(data.name, "The Tower");
        assert_eq!(data.description, Some("A tall stone tower".to_string()));
        assert_eq!(data.loc_type, "building");
    }

    #[test]
    fn test_event_builder() {
        let data = EventBuilder::new("The Betrayal")
            .description("Marcus reveals his true allegiance")
            .sequence(100)
            .build();

        assert_eq!(data.title, "The Betrayal");
        assert_eq!(
            data.description,
            Some("Marcus reveals his true allegiance".to_string())
        );
        assert_eq!(data.sequence, 100);
    }

    #[test]
    fn test_scene_builder() {
        let data = SceneBuilder::new("Opening Scene", "event1", "location1")
            .summary("The story begins")
            .build();

        assert_eq!(data.title, "Opening Scene");
        assert_eq!(data.summary, Some("The story begins".to_string()));
        // RecordId will be "event:event1" and "location:location1"
        assert!(data.event.to_string().contains("event1"));
        assert!(data.primary_location.to_string().contains("location1"));
    }
}
