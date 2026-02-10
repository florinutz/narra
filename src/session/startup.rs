use crate::db::connection::NarraDb;
use crate::error::NarraError;
use crate::session::SessionStateManager;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Determines how verbose the session startup context should be.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StartupVerbosity {
    /// User returned same day - brief reminder
    Brief,
    /// User returned after days - standard context
    Standard,
    /// User returned after weeks - full context
    Full,
    /// First session with existing data
    NewWorld,
    /// No data exists yet
    EmptyWorld,
}

impl std::fmt::Display for StartupVerbosity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartupVerbosity::Brief => write!(f, "brief"),
            StartupVerbosity::Standard => write!(f, "standard"),
            StartupVerbosity::Full => write!(f, "full"),
            StartupVerbosity::NewWorld => write!(f, "new_world"),
            StartupVerbosity::EmptyWorld => write!(f, "empty_world"),
        }
    }
}

/// A hot (recently/frequently accessed) entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotEntity {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub last_accessed: Option<String>,
}

/// Information about a pending decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingDecisionInfo {
    pub id: String,
    pub description: String,
    pub age: String,
    pub affected_count: usize,
}

/// Overview of world entity counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldOverview {
    pub character_count: usize,
    pub location_count: usize,
    pub event_count: usize,
    pub scene_count: usize,
    pub relationship_count: usize,
}

/// Session startup context information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStartupInfo {
    pub verbosity: StartupVerbosity,
    pub last_session_ago: Option<String>,
    pub summary: String,
    pub hot_entities: Vec<HotEntity>,
    pub pending_decisions: Vec<PendingDecisionInfo>,
    pub world_overview: Option<WorldOverview>,
}

/// Generate a human-readable time ago string.
fn format_time_ago(dt: DateTime<Utc>) -> String {
    let now = Utc::now();
    let duration = now.signed_duration_since(dt);

    if duration < Duration::hours(1) {
        let mins = duration.num_minutes();
        if mins <= 1 {
            "just now".to_string()
        } else {
            format!("{} minutes ago", mins)
        }
    } else if duration < Duration::days(1) {
        let hours = duration.num_hours();
        if hours == 1 {
            "1 hour ago".to_string()
        } else {
            format!("{} hours ago", hours)
        }
    } else if duration < Duration::weeks(1) {
        let days = duration.num_days();
        if days == 1 {
            "yesterday".to_string()
        } else {
            format!("{} days ago", days)
        }
    } else if duration < Duration::weeks(4) {
        let weeks = duration.num_weeks();
        if weeks == 1 {
            "1 week ago".to_string()
        } else {
            format!("{} weeks ago", weeks)
        }
    } else {
        let months = duration.num_days() / 30;
        if months == 1 {
            "1 month ago".to_string()
        } else {
            format!("{} months ago", months)
        }
    }
}

/// Query world entity counts from the database.
async fn query_world_overview(db: &NarraDb) -> Result<WorldOverview, NarraError> {
    // Count entities of each type
    let character_count: Option<usize> = db
        .query("SELECT count() FROM character GROUP ALL")
        .await?
        .take((0, "count"))?;

    let location_count: Option<usize> = db
        .query("SELECT count() FROM location GROUP ALL")
        .await?
        .take((0, "count"))?;

    let event_count: Option<usize> = db
        .query("SELECT count() FROM event GROUP ALL")
        .await?
        .take((0, "count"))?;

    let scene_count: Option<usize> = db
        .query("SELECT count() FROM scene GROUP ALL")
        .await?
        .take((0, "count"))?;

    // Count relationships (perceives edges)
    let relationship_count: Option<usize> = db
        .query("SELECT count() FROM perceives GROUP ALL")
        .await?
        .take((0, "count"))?;

    Ok(WorldOverview {
        character_count: character_count.unwrap_or(0),
        location_count: location_count.unwrap_or(0),
        event_count: event_count.unwrap_or(0),
        scene_count: scene_count.unwrap_or(0),
        relationship_count: relationship_count.unwrap_or(0),
    })
}

/// Get entity details for hot entities.
async fn get_hot_entity_details(db: &NarraDb, entity_ids: &[String]) -> Vec<HotEntity> {
    use serde::Deserialize;

    let mut hot_entities = Vec::new();

    #[derive(Deserialize)]
    struct NameOnly {
        name: String,
    }

    for entity_id in entity_ids {
        // Parse table from entity_id (format: "table:id")
        let parts: Vec<&str> = entity_id.split(':').collect();
        if parts.len() != 2 {
            continue;
        }
        let table = parts[0];

        // Query entity name based on table type
        if !matches!(table, "character" | "location" | "event" | "scene") {
            continue;
        }

        // Use direct access (not WHERE id =) per SurrealDB best practices
        let query_result = db.query(format!("SELECT name FROM {}", entity_id)).await;

        if let Ok(mut response) = query_result {
            if let Ok(Some(name_data)) = response.take::<Option<NameOnly>>(0) {
                hot_entities.push(HotEntity {
                    id: entity_id.clone(),
                    name: name_data.name,
                    entity_type: table.to_string(),
                    last_accessed: None, // We don't track timestamp per access, just order
                });
            }
        }
    }

    hot_entities
}

/// Generate session startup context.
pub async fn generate_startup_context(
    session_manager: &SessionStateManager,
    db: &NarraDb,
) -> Result<SessionStartupInfo, NarraError> {
    let last_session = session_manager.get_last_session().await;

    // Get world overview to determine if world has data
    let world_overview = query_world_overview(db).await?;
    let has_data = world_overview.character_count > 0
        || world_overview.location_count > 0
        || world_overview.event_count > 0;

    // Determine verbosity based on time elapsed and data presence
    let verbosity = match last_session {
        None => {
            if has_data {
                StartupVerbosity::NewWorld
            } else {
                StartupVerbosity::EmptyWorld
            }
        }
        Some(last) => {
            let duration = Utc::now().signed_duration_since(last);
            if duration < Duration::hours(24) {
                StartupVerbosity::Brief
            } else if duration < Duration::days(7) {
                StartupVerbosity::Standard
            } else {
                StartupVerbosity::Full
            }
        }
    };

    // Get recent entity accesses
    let recent_limit = match verbosity {
        StartupVerbosity::Brief => 3,
        StartupVerbosity::Standard => 10,
        StartupVerbosity::Full => 20,
        StartupVerbosity::NewWorld => 15,
        StartupVerbosity::EmptyWorld => 0,
    };

    let recent_ids = session_manager.get_recent(recent_limit).await;
    let hot_entities = get_hot_entity_details(db, &recent_ids).await;

    // Get pending decisions
    let pending_decisions_raw = session_manager.get_pending_decisions().await;
    let pending_decisions: Vec<PendingDecisionInfo> = pending_decisions_raw
        .into_iter()
        .map(|d| PendingDecisionInfo {
            id: d.id.clone(),
            description: d.description.clone(),
            age: format_time_ago(d.created_at),
            affected_count: d.entity_ids.len(),
        })
        .collect();

    // Generate summary based on verbosity
    let summary = match verbosity {
        StartupVerbosity::EmptyWorld => {
            "Your Narra world is empty. Ready to start building? Try: 'Create a character named...' or 'Let's establish the setting first.'".to_string()
        }
        StartupVerbosity::NewWorld => {
            let mut parts = Vec::new();
            if world_overview.character_count > 0 {
                parts.push(format!("{} character{}", world_overview.character_count, if world_overview.character_count == 1 { "" } else { "s" }));
            }
            if world_overview.location_count > 0 {
                parts.push(format!("{} location{}", world_overview.location_count, if world_overview.location_count == 1 { "" } else { "s" }));
            }
            if world_overview.event_count > 0 {
                parts.push(format!("{} event{}", world_overview.event_count, if world_overview.event_count == 1 { "" } else { "s" }));
            }

            if parts.is_empty() {
                "Your world is ready. What would you like to create?".to_string()
            } else {
                format!("Your world has {}. Ready to continue.", parts.join(", "))
            }
        }
        StartupVerbosity::Brief => {
            if hot_entities.is_empty() {
                "Welcome back.".to_string()
            } else {
                let names: Vec<&str> = hot_entities.iter().take(3).map(|e| e.name.as_str()).collect();
                format!("Welcome back. You were working with {}.", names.join(", "))
            }
        }
        StartupVerbosity::Standard => {
            let time_ago = last_session.map(format_time_ago).unwrap_or_else(|| "recently".to_string());

            let mut summary_parts = vec![format!("Last session {}", time_ago)];

            if !hot_entities.is_empty() {
                let names: Vec<&str> = hot_entities.iter().take(5).map(|e| e.name.as_str()).collect();
                summary_parts.push(format!("you were working on {}", names.join(", ")));
            }

            if !pending_decisions.is_empty() {
                summary_parts.push(format!("{} pending decision{} need attention",
                    pending_decisions.len(),
                    if pending_decisions.len() == 1 { "" } else { "s" }));
            }

            format!("{}.", summary_parts.join("; "))
        }
        StartupVerbosity::Full => {
            let time_ago = last_session.map(format_time_ago).unwrap_or_else(|| "some time".to_string());

            let mut summary = format!("It's been {} since your last session. ", time_ago);

            if !hot_entities.is_empty() {
                summary.push_str("Here's what you were working on: ");
                let names: Vec<String> = hot_entities.iter()
                    .take(10)
                    .map(|e| format!("{} ({})", e.name, e.entity_type))
                    .collect();
                summary.push_str(&names.join(", "));
                summary.push_str(". ");
            }

            if !pending_decisions.is_empty() {
                summary.push_str(&format!("You have {} pending decision{} that need resolution. ",
                    pending_decisions.len(),
                    if pending_decisions.len() == 1 { "" } else { "s" }));
            }

            summary.push_str("Ready to continue?");
            summary
        }
    };

    let last_session_ago = last_session.map(format_time_ago);
    let overview = if verbosity == StartupVerbosity::NewWorld {
        Some(world_overview)
    } else {
        None
    };

    Ok(SessionStartupInfo {
        verbosity,
        last_session_ago,
        summary,
        hot_entities,
        pending_decisions,
        world_overview: overview,
    })
}
