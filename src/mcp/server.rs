use rmcp::{
    handler::server::tool::ToolRouter,
    handler::server::wrapper::{Json, Parameters},
    model::*,
    service::RequestContext,
    tool, tool_handler, tool_router, ErrorData as McpError, RoleServer, ServerHandler, ServiceExt,
};
use std::sync::Arc;
use surrealdb::{engine::local::Db, Surreal};
use tracing::instrument;

use crate::embedding::{EmbeddingService, StalenessManager};
use crate::mcp::prompts::{
    get_character_voice_prompt, get_conflict_detection_prompt, get_consistency_check_prompt,
    get_consistency_oracle_prompt, get_dramatic_irony_prompt, get_scene_planning_prompt,
};
use crate::mcp::resources::{
    get_consistency_issues_resource, get_entity_resource, get_import_schema, get_import_template,
    get_session_context_resource,
};
use crate::repository::{
    SurrealEntityRepository, SurrealKnowledgeRepository, SurrealRelationshipRepository,
};
use crate::services::{
    CachedContextService, CachedSummaryService, ConsistencyChecker, ConsistencyService,
    ContextService, ImpactAnalyzer, ImpactService, SearchService, SummaryService,
    SurrealSearchService,
};
use crate::session::SessionStateManager;

// Import tool request/response types
use crate::mcp::tools::export::{ExportRequest, ExportResponse};
use crate::mcp::tools::graph::{GraphRequest, GraphResponse};
use crate::mcp::{
    MutationInput, MutationResponse, QueryInput, QueryResponse, SessionInput, SessionResponse,
};

/// MCP server for Narra world state.
///
/// Holds all service dependencies for query and mutation operations.
#[derive(Clone)]
pub struct NarraServer {
    pub(crate) db: Arc<Surreal<Db>>,
    pub(crate) search_service: Arc<dyn SearchService + Send + Sync>,
    pub(crate) context_service: Arc<dyn ContextService + Send + Sync>,
    pub(crate) summary_service: Arc<dyn SummaryService + Send + Sync>,
    pub(crate) impact_service: Arc<dyn ImpactService + Send + Sync>,
    pub(crate) consistency_service: Arc<dyn ConsistencyService>,
    pub(crate) entity_repo: Arc<SurrealEntityRepository>,
    pub(crate) relationship_repo: Arc<SurrealRelationshipRepository>,
    pub(crate) knowledge_repo: Arc<SurrealKnowledgeRepository>,
    pub(crate) session_manager: Arc<SessionStateManager>,
    pub(crate) embedding_service: Arc<dyn EmbeddingService + Send + Sync>,
    pub(crate) staleness_manager: Arc<StalenessManager>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl NarraServer {
    /// Create a new MCP server with the given database connection.
    pub async fn new(
        db: Arc<Surreal<Db>>,
        session_manager: Arc<SessionStateManager>,
        embedding_service: Arc<dyn EmbeddingService + Send + Sync>,
    ) -> Self {
        // Initialize repositories
        let entity_repo = Arc::new(SurrealEntityRepository::new(db.clone()));
        let relationship_repo = Arc::new(SurrealRelationshipRepository::new(db.clone()));
        let knowledge_repo = Arc::new(SurrealKnowledgeRepository::new(db.clone()));

        // Initialize services with repositories
        let search_service: Arc<dyn SearchService + Send + Sync> = Arc::new(
            SurrealSearchService::new(db.clone(), embedding_service.clone()),
        );

        let summary_service: Arc<dyn SummaryService + Send + Sync> =
            Arc::new(CachedSummaryService::with_defaults(db.clone()));

        let context_service: Arc<dyn ContextService + Send + Sync> = Arc::new(
            CachedContextService::new(db.clone(), session_manager.clone()),
        );

        let impact_service: Arc<dyn ImpactService + Send + Sync> =
            Arc::new(ImpactAnalyzer::new(db.clone()));

        let consistency_service: Arc<dyn ConsistencyService> =
            Arc::new(ConsistencyChecker::new(db.clone()));

        // Initialize staleness manager
        let staleness_manager =
            Arc::new(StalenessManager::new(db.clone(), embedding_service.clone()));

        Self {
            db,
            search_service,
            context_service,
            summary_service,
            impact_service,
            consistency_service,
            entity_repo,
            relationship_repo,
            knowledge_repo,
            session_manager,
            embedding_service,
            staleness_manager,
            tool_router: Self::tool_router(),
        }
    }

    // ==========================================================================
    // MCP TOOLS (5 consolidated) - All #[tool] methods must be in this impl block
    // Implementation details are in tools/*.rs files
    // ==========================================================================

    #[tool(
        description = "Query world state: lookup, search (keyword/semantic/hybrid), graph analytics, thematic clustering, arc tracking, perspective vectors, validation, and impact preview. Returns entities with confidence scores."
    )]
    #[instrument(name = "mcp.query", skip_all)]
    pub async fn query(
        &self,
        request: Parameters<QueryInput>,
    ) -> Result<Json<QueryResponse>, String> {
        self.handle_query(request).await.map(Json)
    }

    #[tool(
        description = "Modify world state: create/update/delete entities, record knowledge, manage facts and notes, batch create, import world from YAML (ImportYaml — read narra://schema/import-template first), embeddings, protect entities. CRITICAL violations block mutations."
    )]
    #[instrument(name = "mcp.mutate", skip_all)]
    pub async fn mutate(
        &self,
        request: Parameters<MutationInput>,
    ) -> Result<Json<MutationResponse>, String> {
        self.handle_mutate(request).await.map(Json)
    }

    #[tool(
        description = "Session management: get context summary (hot entities, pinned items, recent work), pin/unpin entities to working context."
    )]
    #[instrument(name = "mcp.session", skip_all)]
    pub async fn session(
        &self,
        request: Parameters<SessionInput>,
    ) -> Result<Json<SessionResponse>, String> {
        self.handle_session(request).await.map(Json)
    }

    #[tool(
        description = "Export all world data to YAML file (NarraImport-compatible). Re-importable with ImportYaml."
    )]
    #[instrument(name = "mcp.export_world", skip_all)]
    pub async fn export_world(
        &self,
        request: Parameters<ExportRequest>,
    ) -> Result<Json<ExportResponse>, String> {
        self.handle_export_world(request).await.map(Json)
    }

    #[tool(
        description = "Generate Mermaid relationship graph to .planning/exports/. Use scope='full' or scope='character:ID' with depth."
    )]
    #[instrument(name = "mcp.generate_graph", skip_all)]
    pub async fn generate_graph(
        &self,
        request: Parameters<GraphRequest>,
    ) -> Result<Json<GraphResponse>, String> {
        self.handle_generate_graph(request).await.map(Json)
    }
}

#[tool_handler]
impl ServerHandler for NarraServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .enable_prompts()
                .build(),
            server_info: Implementation {
                name: "narra".to_string(),
                title: Some("Narra World State Manager".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                icons: None,
                website_url: None,
            },
            instructions: Some(r#"# Narra World State Manager

Narra manages world state for fiction writing: characters, locations, events, scenes, relationships, and character knowledge.

## Tools (5)

**query** — Read-only (40 operations): Lookup, Search, SemanticSearch, HybridSearch, GraphTraversal, Temporal, Overview, ListNotes, GetFact, ListFacts, ReverseQuery, ConnectionPath, CentralityMetrics, InfluencePropagation, DramaticIronyReport, SemanticJoin, ThematicClustering, SemanticKnowledge, SemanticGraphSearch, ArcHistory, ArcComparison, ArcDrift, ArcMoment, PerspectiveSearch, PerceptionGap, PerceptionMatrix, PerceptionShift, UnresolvedTensions, ThematicGaps, SimilarRelationships, KnowledgeConflicts, EmbeddingHealth, WhatIf, ValidateEntity, InvestigateContradictions, KnowledgeAsymmetries, SituationReport, CharacterDossier, ScenePlanning, AnalyzeImpact.

**mutate** — Write (25 operations): CreateCharacter, CreateLocation, CreateEvent, CreateScene, Update, RecordKnowledge, Delete, CreateNote, AttachNote, DetachNote, CreateFact, UpdateFact, DeleteFact, LinkFact, UnlinkFact, CreateRelationship, BatchCreateCharacters, BatchCreateLocations, BatchCreateEvents, BatchCreateRelationships, BackfillEmbeddings, BaselineArcSnapshots, ProtectEntity, UnprotectEntity, ImportYaml.

**session** — Context (3 operations): GetContext, PinEntity, UnpinEntity.

**export** — Export world data to YAML (NarraImport-compatible, re-importable).

**graph** — Generate Mermaid relationship diagram.

## Resources
- `narra://session/context` — Hot entities, pinned items, pending decisions
- `narra://entity/{type}:{id}` — Full entity view (character, location, event, scene)
- `narra://consistency/issues` — Current violations by severity
- `narra://schema/import-template` — YAML template for world import (with examples)
- `narra://schema/import-schema` — JSON Schema for import validation

## Prompts
- `check_consistency` — Guided validation with fix suggestions
- `dramatic_irony` — Knowledge asymmetry analysis

## Key Patterns

**Before deletions**: query(AnalyzeImpact) first, then mutate(Delete).
**Consistency**: CRITICAL violations block mutations. Use query(ValidateEntity) or query(InvestigateContradictions) to debug.
**Semantic search**: Run mutate(BackfillEmbeddings) once after data entry. Use SemanticSearch for concepts, HybridSearch for names+concepts.
**Arc tracking**: Run mutate(BaselineArcSnapshots) once. Then ArcHistory/ArcComparison/ArcDrift/ArcMoment.
**Perspectives**: PerspectiveSearch, PerceptionGap, PerceptionMatrix, PerceptionShift for per-observer views.
**Protection**: mutate(ProtectEntity) marks entities as critical; mutate(UnprotectEntity) removes.
**Session**: session(GetContext) at start; session(PinEntity/UnpinEntity) for persistent context.

## Quick Reference

| Need | Operation |
|------|-----------|
| Find by name | query(Search) |
| Find by meaning | query(SemanticSearch) |
| Check consistency | query(ValidateEntity) |
| Preview deletion | query(AnalyzeImpact) |
| Create character | mutate(CreateCharacter) |
| Record knowledge | mutate(RecordKnowledge) |
| Track evolution | query(ArcHistory) |
| How others see X | query(PerceptionMatrix) |
| Narrative gaps | query(UnresolvedTensions) or query(ThematicGaps) |
| Session context | session(GetContext) |
| Pin entity | session(PinEntity) |
| Protect entity | mutate(ProtectEntity) |
| Import world | mutate(ImportYaml) — read narra://schema/import-template first |
"#.to_string()),
        }
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult {
                resources: vec![
                    Annotated::new(
                        RawResource {
                            uri: "narra://session/context".to_string(),
                            name: "Session Context".to_string(),
                            title: None,
                            description: Some(
                                "Hot entities, recent activity, pinned entities, and pending decisions"
                                    .to_string()
                            ),
                            mime_type: Some("application/json".to_string()),
                            size: None,
                            icons: None,
                            meta: None,
                        },
                        None,
                    ),
                    Annotated::new(
                        RawResource {
                            uri: "narra://consistency/issues".to_string(),
                            name: "Consistency Issues".to_string(),
                            title: None,
                            description: Some(
                                "Current consistency violations grouped by severity (Critical/Warning/Info)"
                                    .to_string()
                            ),
                            mime_type: Some("application/json".to_string()),
                            size: None,
                            icons: None,
                            meta: None,
                        },
                        None,
                    ),
                    Annotated::new(
                        RawResource {
                            uri: "narra://schema/import-template".to_string(),
                            name: "Import Template".to_string(),
                            title: None,
                            description: Some(
                                "YAML template with all 8 entity types, realistic examples, and inline comments for world import"
                                    .to_string()
                            ),
                            mime_type: Some("text/yaml".to_string()),
                            size: None,
                            icons: None,
                            meta: None,
                        },
                        None,
                    ),
                    Annotated::new(
                        RawResource {
                            uri: "narra://schema/import-schema".to_string(),
                            name: "Import JSON Schema".to_string(),
                            title: None,
                            description: Some(
                                "Auto-generated JSON Schema for the NarraImport type, useful for validation"
                                    .to_string()
                            ),
                            mime_type: Some("application/schema+json".to_string()),
                            size: None,
                            icons: None,
                            meta: None,
                        },
                        None,
                    ),
                    Annotated::new(
                        RawResource {
                            uri: "narra://analysis/tension-matrix".to_string(),
                            name: "Tension Matrix".to_string(),
                            title: None,
                            description: Some(
                                "All character pairs with tension scores, showing the emotional landscape"
                                    .to_string()
                            ),
                            mime_type: Some("application/json".to_string()),
                            size: None,
                            icons: None,
                            meta: None,
                        },
                        None,
                    ),
                ],
                next_cursor: None,
                meta: None,
            })
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult {
                resource_templates: vec![
                    Annotated::new(
                        RawResourceTemplate {
                            uri_template: "narra://entity/{type}:{id}".to_string(),
                            name: "Entity View".to_string(),
                            title: None,
                            description: Some(
                                "Full entity view with attributes, relationships, and knowledge. \
                                 Types: character, location, event, scene. Example: narra://entity/character:alice"
                                    .to_string()
                            ),
                            mime_type: Some("application/json".to_string()),
                            icons: None,
                        },
                        None,
                    ),
                    Annotated::new(
                        RawResourceTemplate {
                            uri_template: "narra://character/{id}/dossier".to_string(),
                            name: "Character Dossier".to_string(),
                            title: None,
                            description: Some(
                                "Comprehensive character analysis: network position, knowledge, perceptions. \
                                 Example: narra://character/alice/dossier"
                                    .to_string()
                            ),
                            mime_type: Some("application/json".to_string()),
                            icons: None,
                        },
                        None,
                    ),
                ],
                next_cursor: None,
                meta: None,
            })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        let uri = &request.uri;

        // Validate scheme
        if !uri.starts_with("narra://") {
            return Err(McpError::resource_not_found(
                format!("Invalid URI scheme, expected narra://: {}", uri),
                None,
            ));
        }

        // Route based on URI path
        if uri == "narra://session/context" {
            self.read_session_context_resource(uri).await
        } else if uri == "narra://consistency/issues" {
            self.read_consistency_issues_resource(uri).await
        } else if uri == "narra://schema/import-template" {
            self.read_import_template_resource(uri)
        } else if uri == "narra://schema/import-schema" {
            self.read_import_schema_resource(uri)
        } else if uri == "narra://analysis/tension-matrix" {
            self.read_tension_matrix_resource(uri).await
        } else if let Some(rest) = uri.strip_prefix("narra://character/") {
            if let Some(char_id) = rest.strip_suffix("/dossier") {
                let full_id = format!("character:{}", char_id);
                self.read_character_dossier_resource(uri, &full_id).await
            } else {
                Err(McpError::resource_not_found(
                    format!("Unknown resource: {}", uri),
                    None,
                ))
            }
        } else if let Some(entity_id) = uri.strip_prefix("narra://entity/") {
            self.read_entity_resource(uri, entity_id).await
        } else {
            Err(McpError::resource_not_found(
                format!("Unknown resource: {}", uri),
                None,
            ))
        }
    }

    async fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, McpError> {
        Ok(ListPromptsResult {
            prompts: vec![
                Prompt::new(
                    "check_consistency",
                    Some("Validate entities for consistency issues and suggest fixes"),
                    Some(vec![
                        PromptArgument {
                            name: "entity_id".into(),
                            title: None,
                            description: Some(
                                "Focus on specific entity (e.g., character:alice)".into(),
                            ),
                            required: Some(false),
                        },
                        PromptArgument {
                            name: "severity_filter".into(),
                            title: None,
                            description: Some(
                                "Minimum severity: critical, warning, or info".into(),
                            ),
                            required: Some(false),
                        },
                    ]),
                ),
                Prompt::new(
                    "dramatic_irony",
                    Some("Analyze knowledge asymmetry between characters for dramatic tension"),
                    Some(vec![
                        PromptArgument {
                            name: "character_a".into(),
                            title: None,
                            description: Some("First character to compare".into()),
                            required: Some(true),
                        },
                        PromptArgument {
                            name: "character_b".into(),
                            title: None,
                            description: Some("Second character (or compare against all)".into()),
                            required: Some(false),
                        },
                        PromptArgument {
                            name: "topic".into(),
                            title: None,
                            description: Some("Focus on knowledge about specific entity".into()),
                            required: Some(false),
                        },
                    ]),
                ),
                Prompt::new(
                    "scene_planning",
                    Some("Guided scene construction with pairwise dynamics, irony, and facts"),
                    Some(vec![
                        PromptArgument {
                            name: "character_ids".into(),
                            title: None,
                            description: Some(
                                "Comma-separated character IDs (e.g., character:alice,character:bob)"
                                    .into(),
                            ),
                            required: Some(true),
                        },
                        PromptArgument {
                            name: "location_id".into(),
                            title: None,
                            description: Some("Location for the scene".into()),
                            required: Some(false),
                        },
                        PromptArgument {
                            name: "tone".into(),
                            title: None,
                            description: Some(
                                "Desired tone (e.g., tense, comedic, intimate)".into(),
                            ),
                            required: Some(false),
                        },
                    ]),
                ),
                Prompt::new(
                    "character_voice",
                    Some("Build a voice profile for consistent character dialogue"),
                    Some(vec![
                        PromptArgument {
                            name: "character_id".into(),
                            title: None,
                            description: Some("Character ID (e.g., character:alice)".into()),
                            required: Some(true),
                        },
                        PromptArgument {
                            name: "context".into(),
                            title: None,
                            description: Some(
                                "Scene or event context for the dialogue".into(),
                            ),
                            required: Some(false),
                        },
                    ]),
                ),
                Prompt::new(
                    "conflict_detection",
                    Some("Identify active and latent conflicts with escalation/resolution paths"),
                    Some(vec![PromptArgument {
                        name: "scope".into(),
                        title: None,
                        description: Some(
                            "\"all\" for world-wide, or entity ID to focus on".into(),
                        ),
                        required: Some(false),
                    }]),
                ),
                Prompt::new(
                    "world_consistency_oracle",
                    Some("Validate a proposed plot action against established world state"),
                    Some(vec![
                        PromptArgument {
                            name: "proposed_action".into(),
                            title: None,
                            description: Some(
                                "Natural language description of the proposed action".into(),
                            ),
                            required: Some(true),
                        },
                        PromptArgument {
                            name: "affected_entities".into(),
                            title: None,
                            description: Some(
                                "Comma-separated entity IDs that might be affected".into(),
                            ),
                            required: Some(false),
                        },
                    ]),
                ),
            ],
            next_cursor: None,
            meta: None,
        })
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, McpError> {
        match request.name.as_str() {
            "check_consistency" => Ok(get_consistency_check_prompt(request.arguments)),
            "dramatic_irony" => Ok(get_dramatic_irony_prompt(request.arguments)),
            "scene_planning" => Ok(get_scene_planning_prompt(request.arguments)),
            "character_voice" => Ok(get_character_voice_prompt(request.arguments)),
            "conflict_detection" => Ok(get_conflict_detection_prompt(request.arguments)),
            "world_consistency_oracle" => Ok(get_consistency_oracle_prompt(request.arguments)),
            _ => Err(McpError::invalid_params(
                format!("Unknown prompt: {}", request.name),
                None,
            )),
        }
    }
}

impl NarraServer {
    /// Create server from shared AppContext (used by unified binary).
    pub async fn from_context(ctx: &crate::init::AppContext) -> Self {
        Self {
            db: ctx.db.clone(),
            search_service: ctx.search_service.clone(),
            context_service: ctx.context_service.clone(),
            summary_service: ctx.summary_service.clone(),
            impact_service: ctx.impact_service.clone(),
            consistency_service: ctx.consistency_service.clone(),
            entity_repo: ctx.entity_repo.clone(),
            relationship_repo: ctx.relationship_repo.clone(),
            knowledge_repo: ctx.knowledge_repo.clone(),
            session_manager: ctx.session_manager.clone(),
            embedding_service: ctx.embedding_service.clone(),
            staleness_manager: ctx.staleness_manager.clone(),
            tool_router: Self::tool_router(),
        }
    }

    async fn read_session_context_resource(
        &self,
        uri: &str,
    ) -> Result<ReadResourceResult, McpError> {
        let content = get_session_context_resource(&self.session_manager, &self.db)
            .await
            .map_err(|e| McpError::internal_error(e, None))?;

        Ok(ReadResourceResult {
            contents: vec![ResourceContents::TextResourceContents {
                uri: uri.to_string(),
                mime_type: Some("application/json".to_string()),
                text: content,
                meta: None,
            }],
        })
    }

    async fn read_consistency_issues_resource(
        &self,
        uri: &str,
    ) -> Result<ReadResourceResult, McpError> {
        let content = get_consistency_issues_resource(&self.db, &self.consistency_service)
            .await
            .map_err(|e| McpError::internal_error(e, None))?;

        Ok(ReadResourceResult {
            contents: vec![ResourceContents::TextResourceContents {
                uri: uri.to_string(),
                mime_type: Some("application/json".to_string()),
                text: content,
                meta: None,
            }],
        })
    }

    fn read_import_template_resource(&self, uri: &str) -> Result<ReadResourceResult, McpError> {
        Ok(ReadResourceResult {
            contents: vec![ResourceContents::TextResourceContents {
                uri: uri.to_string(),
                mime_type: Some("text/yaml".to_string()),
                text: get_import_template(),
                meta: None,
            }],
        })
    }

    fn read_import_schema_resource(&self, uri: &str) -> Result<ReadResourceResult, McpError> {
        Ok(ReadResourceResult {
            contents: vec![ResourceContents::TextResourceContents {
                uri: uri.to_string(),
                mime_type: Some("application/schema+json".to_string()),
                text: get_import_schema(),
                meta: None,
            }],
        })
    }

    async fn read_entity_resource(
        &self,
        uri: &str,
        entity_id: &str,
    ) -> Result<ReadResourceResult, McpError> {
        // Validate entity_id format (should be type:id)
        if !entity_id.contains(':') {
            return Err(McpError::resource_not_found(
                format!(
                    "Invalid entity ID format, expected type:id, got: {}",
                    entity_id
                ),
                None,
            ));
        }

        let content = get_entity_resource(entity_id, &self.context_service)
            .await
            .map_err(|e| {
                if e.contains("not found") {
                    McpError::resource_not_found(e, None)
                } else {
                    McpError::internal_error(e, None)
                }
            })?;

        Ok(ReadResourceResult {
            contents: vec![ResourceContents::TextResourceContents {
                uri: uri.to_string(),
                mime_type: Some("application/json".to_string()),
                text: content,
                meta: None,
            }],
        })
    }

    async fn read_tension_matrix_resource(
        &self,
        uri: &str,
    ) -> Result<ReadResourceResult, McpError> {
        let response = self
            .handle_tension_matrix(Some(1), Some(50))
            .await
            .map_err(|e| McpError::internal_error(e, None))?;

        let content = response
            .results
            .first()
            .map(|r| r.content.clone())
            .unwrap_or_else(|| "No tension data".to_string());

        Ok(ReadResourceResult {
            contents: vec![ResourceContents::TextResourceContents {
                uri: uri.to_string(),
                mime_type: Some("text/markdown".to_string()),
                text: content,
                meta: None,
            }],
        })
    }

    async fn read_character_dossier_resource(
        &self,
        uri: &str,
        character_id: &str,
    ) -> Result<ReadResourceResult, McpError> {
        let response = self
            .handle_character_dossier(character_id)
            .await
            .map_err(|e| {
                if e.contains("not found") {
                    McpError::resource_not_found(e, None)
                } else {
                    McpError::internal_error(e, None)
                }
            })?;

        let content = response
            .results
            .first()
            .map(|r| r.content.clone())
            .unwrap_or_else(|| "No dossier data".to_string());

        Ok(ReadResourceResult {
            contents: vec![ResourceContents::TextResourceContents {
                uri: uri.to_string(),
                mime_type: Some("text/markdown".to_string()),
                text: content,
                meta: None,
            }],
        })
    }
}

/// Run MCP server on stdio transport.
pub async fn run_mcp_server(ctx: crate::init::AppContext) -> anyhow::Result<()> {
    let server = NarraServer::from_context(&ctx).await;

    tracing::info!("Starting Narra MCP server v{}", env!("CARGO_PKG_VERSION"));

    // Measure context budget
    {
        fn estimate_tokens(text: &str) -> usize {
            text.len().div_ceil(4)
        }

        let server_info = server.get_info();
        let info_json = serde_json::to_string(&server_info).unwrap_or_default();
        let info_tokens = estimate_tokens(&info_json);
        let instructions_tokens = server_info
            .instructions
            .as_ref()
            .map(|i| estimate_tokens(i))
            .unwrap_or(0);
        let total_tokens = info_tokens + instructions_tokens;

        tracing::info!(
            "MCP context budget: {} tokens (info: {}, instructions: {})",
            total_tokens,
            info_tokens,
            instructions_tokens
        );

        if total_tokens > 8_000 {
            tracing::warn!("Context budget exceeds 8K tokens ({})", total_tokens);
        }
    }

    // Stdio transport
    let transport = (tokio::io::stdin(), tokio::io::stdout());
    let service = server.serve(transport).await?;
    tracing::info!("MCP server listening on stdio (5 tools)");

    // Graceful shutdown
    let session_manager = ctx.session_manager.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Shutdown signal received");
        session_manager.mark_session_end().await;
        if let Err(e) = session_manager.save().await {
            tracing::error!("Failed to save session state: {}", e);
        }
    });

    service.waiting().await?;

    tracing::info!("MCP server shutting down");
    ctx.session_manager.mark_session_end().await;
    ctx.session_manager.save().await?;
    tracing::info!("Session state saved");

    Ok(())
}
