use crate::models::{
    Character, CharacterCreate, CharacterUpdate, Event, EventCreate, EventUpdate, Location,
    LocationCreate, LocationUpdate, Scene, SceneCreate, SceneUpdate,
};
use crate::NarraError;
use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use std::sync::Arc;
use surrealdb::engine::local::Db;
use surrealdb::Surreal;

/// Repository trait for core entity CRUD operations.
///
/// Covers: Character, Location, Event, Scene
/// Each entity type has get, list, create, update, delete operations.
#[async_trait]
pub trait EntityRepository: Send + Sync {
    // Character operations
    async fn get_character(&self, id: &str) -> Result<Option<Character>, NarraError>;
    async fn list_characters(&self) -> Result<Vec<Character>, NarraError>;
    async fn create_character(&self, data: CharacterCreate) -> Result<Character, NarraError>;
    async fn update_character(
        &self,
        id: &str,
        data: CharacterUpdate,
    ) -> Result<Option<Character>, NarraError>;
    async fn delete_character(&self, id: &str) -> Result<Option<Character>, NarraError>;

    // Location operations
    async fn get_location(&self, id: &str) -> Result<Option<Location>, NarraError>;
    async fn list_locations(&self) -> Result<Vec<Location>, NarraError>;
    async fn create_location(&self, data: LocationCreate) -> Result<Location, NarraError>;
    async fn update_location(
        &self,
        id: &str,
        data: LocationUpdate,
    ) -> Result<Option<Location>, NarraError>;
    async fn delete_location(&self, id: &str) -> Result<Option<Location>, NarraError>;

    // Event operations
    async fn get_event(&self, id: &str) -> Result<Option<Event>, NarraError>;
    async fn list_events(&self) -> Result<Vec<Event>, NarraError>;
    async fn create_event(&self, data: EventCreate) -> Result<Event, NarraError>;
    async fn update_event(&self, id: &str, data: EventUpdate) -> Result<Option<Event>, NarraError>;
    async fn delete_event(&self, id: &str) -> Result<Option<Event>, NarraError>;

    // Scene operations
    async fn get_scene(&self, id: &str) -> Result<Option<Scene>, NarraError>;
    async fn list_scenes(&self) -> Result<Vec<Scene>, NarraError>;
    async fn create_scene(&self, data: SceneCreate) -> Result<Scene, NarraError>;
    async fn update_scene(&self, id: &str, data: SceneUpdate) -> Result<Option<Scene>, NarraError>;
    async fn delete_scene(&self, id: &str) -> Result<Option<Scene>, NarraError>;

    // Batch operations
    async fn create_characters_batch<I>(&self, items: I) -> Result<Vec<Character>, NarraError>
    where
        I: IntoIterator<Item = CharacterCreate> + Send,
        I::IntoIter: Send;
}

/// SurrealDB implementation of EntityRepository.
///
/// Wraps the database connection and delegates to model functions.
pub struct SurrealEntityRepository {
    db: Arc<Surreal<Db>>,
}

impl SurrealEntityRepository {
    /// Create a new repository with the given database connection.
    pub fn new(db: Arc<Surreal<Db>>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl EntityRepository for SurrealEntityRepository {
    // Character operations - delegate to models::character functions
    async fn get_character(&self, id: &str) -> Result<Option<Character>, NarraError> {
        crate::models::character::get_character(&self.db, id).await
    }

    async fn list_characters(&self) -> Result<Vec<Character>, NarraError> {
        crate::models::character::list_characters(&self.db).await
    }

    async fn create_character(&self, data: CharacterCreate) -> Result<Character, NarraError> {
        crate::models::character::create_character(&self.db, data).await
    }

    async fn update_character(
        &self,
        id: &str,
        data: CharacterUpdate,
    ) -> Result<Option<Character>, NarraError> {
        crate::models::character::update_character(&self.db, id, data).await
    }

    async fn delete_character(&self, id: &str) -> Result<Option<Character>, NarraError> {
        crate::models::character::delete_character(&self.db, id).await
    }

    // Location operations - delegate to models::location functions
    async fn get_location(&self, id: &str) -> Result<Option<Location>, NarraError> {
        crate::models::location::get_location(&self.db, id).await
    }

    async fn list_locations(&self) -> Result<Vec<Location>, NarraError> {
        crate::models::location::list_locations(&self.db).await
    }

    async fn create_location(&self, data: LocationCreate) -> Result<Location, NarraError> {
        crate::models::location::create_location(&self.db, data).await
    }

    async fn update_location(
        &self,
        id: &str,
        data: LocationUpdate,
    ) -> Result<Option<Location>, NarraError> {
        crate::models::location::update_location(&self.db, id, data).await
    }

    async fn delete_location(&self, id: &str) -> Result<Option<Location>, NarraError> {
        crate::models::location::delete_location(&self.db, id).await
    }

    // Event operations - delegate to models::event functions
    async fn get_event(&self, id: &str) -> Result<Option<Event>, NarraError> {
        crate::models::event::get_event(&self.db, id).await
    }

    async fn list_events(&self) -> Result<Vec<Event>, NarraError> {
        crate::models::event::list_events_ordered(&self.db).await
    }

    async fn create_event(&self, data: EventCreate) -> Result<Event, NarraError> {
        crate::models::event::create_event(&self.db, data).await
    }

    async fn update_event(&self, id: &str, data: EventUpdate) -> Result<Option<Event>, NarraError> {
        crate::models::event::update_event(&self.db, id, data).await
    }

    async fn delete_event(&self, id: &str) -> Result<Option<Event>, NarraError> {
        crate::models::event::delete_event(&self.db, id).await
    }

    // Scene operations - delegate to models::scene functions
    async fn get_scene(&self, id: &str) -> Result<Option<Scene>, NarraError> {
        crate::models::scene::get_scene(&self.db, id).await
    }

    async fn list_scenes(&self) -> Result<Vec<Scene>, NarraError> {
        crate::models::scene::list_scenes(&self.db).await
    }

    async fn create_scene(&self, data: SceneCreate) -> Result<Scene, NarraError> {
        crate::models::scene::create_scene(&self.db, data).await
    }

    async fn update_scene(&self, id: &str, data: SceneUpdate) -> Result<Option<Scene>, NarraError> {
        crate::models::scene::update_scene(&self.db, id, data).await
    }

    async fn delete_scene(&self, id: &str) -> Result<Option<Scene>, NarraError> {
        crate::models::scene::delete_scene(&self.db, id).await
    }

    // Batch operations
    async fn create_characters_batch<I>(&self, items: I) -> Result<Vec<Character>, NarraError>
    where
        I: IntoIterator<Item = CharacterCreate> + Send,
        I::IntoIter: Send,
    {
        let mut results = Vec::new();
        let chunks = stream::iter(items).chunks(50); // Process 50 at a time
        tokio::pin!(chunks);

        while let Some(chunk) = chunks.next().await {
            for item in chunk {
                let character = crate::models::character::create_character(&self.db, item).await?;
                results.push(character);
            }
        }

        Ok(results)
    }
}
