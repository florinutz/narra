use crate::mcp::NarraServer;
use crate::session::{generate_startup_context, HotEntity, PendingDecisionInfo, WorldOverview};
use rmcp::handler::server::wrapper::Parameters;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Request for session context.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SessionContextRequest {
    /// Force full verbosity regardless of time elapsed
    #[serde(default)]
    pub force_full: bool,
}

/// Response structure for hot entity.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HotEntityResponse {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub last_accessed: Option<String>,
}

impl From<HotEntity> for HotEntityResponse {
    fn from(entity: HotEntity) -> Self {
        Self {
            id: entity.id,
            name: entity.name,
            entity_type: entity.entity_type,
            last_accessed: entity.last_accessed,
        }
    }
}

/// Response structure for pending decision.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PendingDecisionResponse {
    pub id: String,
    pub description: String,
    pub age: String,
    pub affected_count: usize,
}

impl From<PendingDecisionInfo> for PendingDecisionResponse {
    fn from(decision: PendingDecisionInfo) -> Self {
        Self {
            id: decision.id,
            description: decision.description,
            age: decision.age,
            affected_count: decision.affected_count,
        }
    }
}

/// Response structure for world overview.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WorldOverviewResponse {
    pub character_count: usize,
    pub location_count: usize,
    pub event_count: usize,
    pub scene_count: usize,
    pub relationship_count: usize,
}

impl From<WorldOverview> for WorldOverviewResponse {
    fn from(overview: WorldOverview) -> Self {
        Self {
            character_count: overview.character_count,
            location_count: overview.location_count,
            event_count: overview.event_count,
            scene_count: overview.scene_count,
            relationship_count: overview.relationship_count,
        }
    }
}

/// Response containing session context.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SessionContextResponse {
    pub verbosity: String,
    pub summary: String,
    pub last_session_ago: Option<String>,
    pub hot_entities: Vec<HotEntityResponse>,
    pub pending_decisions: Vec<PendingDecisionResponse>,
    pub world_overview: Option<WorldOverviewResponse>,
}

impl NarraServer {
    /// Handler for get_session_context tool - implementation called from server.rs
    pub async fn handle_get_session_context(
        &self,
        Parameters(_request): Parameters<SessionContextRequest>,
    ) -> Result<SessionContextResponse, String> {
        // Generate startup context
        let startup_info = generate_startup_context(&self.session_manager, &self.db)
            .await
            .map_err(|e| format!("Failed to generate session context: {}", e))?;

        // Convert to response format
        Ok(SessionContextResponse {
            verbosity: startup_info.verbosity.to_string(),
            summary: startup_info.summary,
            last_session_ago: startup_info.last_session_ago,
            hot_entities: startup_info
                .hot_entities
                .into_iter()
                .map(Into::into)
                .collect(),
            pending_decisions: startup_info
                .pending_decisions
                .into_iter()
                .map(Into::into)
                .collect(),
            world_overview: startup_info.world_overview.map(Into::into),
        })
    }
}
