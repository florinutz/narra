use crate::db::connection::NarraDb;
use crate::mcp::progress::make_mcp_progress;
use rmcp::{
    handler::server::tool::ToolRouter,
    handler::server::wrapper::{Json, Parameters},
    model::*,
    service::RequestContext,
    tool, tool_handler, tool_router, ErrorData as McpError, Peer, RoleServer, ServerHandler,
    ServiceExt,
};
use std::sync::Arc;
use tracing::instrument;

use crate::embedding::{EmbeddingService, StalenessManager};
use crate::mcp::prompts::{
    get_character_voice_prompt, get_conflict_detection_prompt, get_consistency_check_prompt,
    get_consistency_oracle_prompt, get_dramatic_irony_prompt, get_getting_started_prompt,
    get_scene_planning_prompt,
};
use crate::mcp::resources::{
    get_consistency_issues_resource, get_entity_resource, get_import_schema, get_import_template,
    get_operations_guide, get_session_context_resource, get_world_overview_resource,
};
use crate::repository::{
    SurrealEntityRepository, SurrealKnowledgeRepository, SurrealRelationshipRepository,
};
use crate::services::EmotionService;
use crate::services::NerService;
use crate::services::ThemeService;
use crate::services::{
    CachedContextService, CachedSummaryService, ConsistencyChecker, ConsistencyService,
    ContextService, ImpactAnalyzer, ImpactService, SearchService, SummaryService,
    SurrealSearchService,
};
use crate::session::SessionStateManager;

// Import tool request/response types
use crate::mcp::error::ToolError;
use crate::mcp::tools::export::{ExportRequest, ExportResponse};
use crate::mcp::tools::graph::{GraphRequest, GraphResponse};
use crate::mcp::{
    CreateCharacterInput, CreateRelationshipInput, DetailLevel, DossierInput, IronyReportInput,
    KeywordSearchInput, KnowledgeAsymmetriesInput, LookupInput, MutationInput, MutationResponse,
    OverviewInput, QueryInput, QueryResponse, RecordKnowledgeInput, ScenePrepInput,
    SemanticSearchInput, SessionInput, SessionResponse, UpdateEntityInput, ValidateEntityInput,
    DEFAULT_TOKEN_BUDGET, MAX_LIMIT,
};

/// MCP server for Narra world state.
///
/// Holds all service dependencies for query and mutation operations.
#[derive(Clone)]
pub struct NarraServer {
    pub(crate) db: Arc<NarraDb>,
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
    pub(crate) emotion_service: Arc<dyn EmotionService + Send + Sync>,
    pub(crate) theme_service: Arc<dyn ThemeService + Send + Sync>,
    pub(crate) ner_service: Arc<dyn NerService + Send + Sync>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl NarraServer {
    /// Create a new MCP server with the given database connection.
    pub async fn new(
        db: Arc<NarraDb>,
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

        // Emotion classifier — degrades gracefully if model unavailable
        let emotion_service: Arc<dyn EmotionService + Send + Sync> = {
            let service = crate::services::LocalEmotionService::new(db.clone());
            if service.is_available() {
                Arc::new(service)
            } else {
                Arc::new(crate::services::NoopEmotionService::new())
            }
        };

        // Theme classifier — degrades gracefully if model unavailable
        let theme_service: Arc<dyn ThemeService + Send + Sync> = {
            let service = crate::services::LocalThemeService::new(db.clone());
            if service.is_available() {
                Arc::new(service)
            } else {
                Arc::new(crate::services::NoopThemeService::new())
            }
        };

        // NER classifier — degrades gracefully if model unavailable
        let ner_service: Arc<dyn NerService + Send + Sync> = {
            let service = crate::services::LocalNerService::new(db.clone());
            if service.is_available() {
                Arc::new(service)
            } else {
                Arc::new(crate::services::NoopNerService::new())
            }
        };

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
            emotion_service,
            theme_service,
            ner_service,
            tool_router: Self::tool_router(),
        }
    }

    // ==========================================================================
    // MCP TOOLS (18 total) - All #[tool] methods must be in this impl block
    // 5 parameterized + 13 dedicated. Implementation details in tools/*.rs files
    // ==========================================================================

    #[tool(
        description = "Advanced read operations (40): graph analytics, arc tracking, perspectives, clustering, vector ops, and more. For common queries, prefer the dedicated tools: semantic_search, search, lookup, dossier, scene_prep, irony_report, knowledge_asymmetries, validate_entity."
    )]
    #[instrument(name = "mcp.query", skip_all)]
    pub async fn query(
        &self,
        request: Parameters<QueryInput>,
        meta: Meta,
        client: Peer<RoleServer>,
    ) -> Result<Json<QueryResponse>, ToolError> {
        let progress = make_mcp_progress(&meta, &client);
        progress
            .report(0.0, 1.0, Some("Processing query...".into()))
            .await;

        let result = self
            .handle_query(request)
            .await
            .map(Json)
            .map_err(ToolError::from);

        progress
            .report(1.0, 1.0, Some("Query complete".into()))
            .await;
        result
    }

    #[tool(
        description = "Advanced write operations (25): batch creation, YAML import, embeddings, arc baselines, protect/unprotect, and more. For common writes, prefer: record_knowledge, create_character, create_relationship, update_entity."
    )]
    #[instrument(name = "mcp.mutate", skip_all)]
    pub async fn mutate(
        &self,
        request: Parameters<MutationInput>,
        meta: Meta,
        client: Peer<RoleServer>,
    ) -> Result<Json<MutationResponse>, ToolError> {
        let progress = make_mcp_progress(&meta, &client);
        progress
            .report(0.0, 1.0, Some("Processing mutation...".into()))
            .await;

        let result = self
            .handle_mutate(request)
            .await
            .map(Json)
            .map_err(ToolError::from);

        progress
            .report(1.0, 1.0, Some("Mutation complete".into()))
            .await;
        result
    }

    #[tool(
        description = "Session management: get context summary (hot entities, pinned items, recent work), pin/unpin entities to working context."
    )]
    #[instrument(name = "mcp.session", skip_all)]
    pub async fn session(
        &self,
        request: Parameters<SessionInput>,
    ) -> Result<Json<SessionResponse>, ToolError> {
        self.handle_session(request)
            .await
            .map(Json)
            .map_err(ToolError::from)
    }

    #[tool(
        description = "Export all world data to YAML file (NarraImport-compatible). Re-importable with ImportYaml."
    )]
    #[instrument(name = "mcp.export_world", skip_all)]
    pub async fn export_world(
        &self,
        request: Parameters<ExportRequest>,
    ) -> Result<Json<ExportResponse>, ToolError> {
        self.handle_export_world(request)
            .await
            .map(Json)
            .map_err(ToolError::from)
    }

    #[tool(
        description = "Generate Mermaid relationship graph to .planning/exports/. Use scope='full' or scope='character:ID' with depth."
    )]
    #[instrument(name = "mcp.generate_graph", skip_all)]
    pub async fn generate_graph(
        &self,
        request: Parameters<GraphRequest>,
    ) -> Result<Json<GraphResponse>, ToolError> {
        self.handle_generate_graph(request)
            .await
            .map(Json)
            .map_err(ToolError::from)
    }

    // ==========================================================================
    // DEDICATED TOOLS — Essential 5
    // ==========================================================================

    #[tool(
        description = "Find entities by meaning and theme using semantic similarity. Best for concept queries like 'characters struggling with duty' or 'scenes about betrayal'. Use 'search' for keyword/name lookups instead."
    )]
    #[instrument(name = "mcp.semantic_search", skip_all)]
    pub async fn semantic_search(
        &self,
        request: Parameters<SemanticSearchInput>,
    ) -> Result<Json<QueryResponse>, ToolError> {
        let Parameters(input) = request;
        self.handle_unified_search(
            &input.query,
            "semantic",
            input.entity_types,
            input.limit.unwrap_or(10).min(MAX_LIMIT),
            None,
            None,
        )
        .await
        .map(Json)
        .map_err(ToolError::from)
    }

    #[tool(
        description = "Full character analysis: network position, knowledge, perceptions, arc trajectory, and narrative suggestions. Takes a character ID."
    )]
    #[instrument(name = "mcp.dossier", skip_all)]
    pub async fn dossier(
        &self,
        request: Parameters<DossierInput>,
        meta: Meta,
        client: Peer<RoleServer>,
    ) -> Result<Json<QueryResponse>, ToolError> {
        let progress = make_mcp_progress(&meta, &client);
        let Parameters(input) = request;
        self.handle_character_dossier(&input.character_id, None, DEFAULT_TOKEN_BUDGET, progress)
            .await
            .map(Json)
            .map_err(ToolError::from)
    }

    #[tool(
        description = "Scene preparation: pairwise character dynamics, dramatic irony opportunities, tensions, and applicable facts for a set of characters about to meet."
    )]
    #[instrument(name = "mcp.scene_prep", skip_all)]
    pub async fn scene_prep(
        &self,
        request: Parameters<ScenePrepInput>,
        meta: Meta,
        client: Peer<RoleServer>,
    ) -> Result<Json<QueryResponse>, ToolError> {
        let progress = make_mcp_progress(&meta, &client);
        let Parameters(input) = request;
        self.handle_scene_planning(&input.character_ids, None, DEFAULT_TOKEN_BUDGET, progress)
            .await
            .map(Json)
            .map_err(ToolError::from)
    }

    #[tool(
        description = "World overview: entity counts and summaries. Filter by type ('character', 'location', 'event', 'scene') or 'all'."
    )]
    #[instrument(name = "mcp.overview", skip_all)]
    pub async fn overview(
        &self,
        request: Parameters<OverviewInput>,
        meta: Meta,
        client: Peer<RoleServer>,
    ) -> Result<Json<QueryResponse>, ToolError> {
        let progress = make_mcp_progress(&meta, &client);
        let Parameters(input) = request;
        self.handle_overview(
            &input.entity_type,
            input.limit.unwrap_or(20).min(MAX_LIMIT),
            progress,
        )
        .await
        .map(Json)
        .map_err(ToolError::from)
    }

    #[tool(
        description = "Record what a character knows or believes about something. Supports certainty levels: knows, suspects, believes_wrongly, uncertain."
    )]
    #[instrument(name = "mcp.record_knowledge", skip_all)]
    pub async fn record_knowledge(
        &self,
        request: Parameters<RecordKnowledgeInput>,
    ) -> Result<Json<MutationResponse>, ToolError> {
        let Parameters(input) = request;
        self.handle_record_knowledge(
            input.character_id,
            input.target_id,
            input.fact,
            input.certainty,
            input.method,
            input.source_character_id,
            input.event_id,
        )
        .await
        .map(Json)
        .map_err(ToolError::from)
    }

    // ==========================================================================
    // DEDICATED TOOLS — Standard 8
    // ==========================================================================

    #[tool(
        description = "Find entities by keyword (names, titles). Use 'semantic_search' for concept/theme queries instead."
    )]
    #[instrument(name = "mcp.search", skip_all)]
    pub async fn search(
        &self,
        request: Parameters<KeywordSearchInput>,
    ) -> Result<Json<QueryResponse>, ToolError> {
        let Parameters(input) = request;
        self.handle_search(
            &input.query,
            input.entity_types,
            input.limit.unwrap_or(10).min(MAX_LIMIT),
            None,
        )
        .await
        .map(Json)
        .map_err(ToolError::from)
    }

    #[tool(
        description = "Get a specific entity by ID. Returns full entity data with relationships and knowledge."
    )]
    #[instrument(name = "mcp.lookup", skip_all)]
    pub async fn lookup(
        &self,
        request: Parameters<LookupInput>,
    ) -> Result<Json<QueryResponse>, ToolError> {
        let Parameters(input) = request;
        let detail_level = match input.detail_level.as_deref() {
            Some("full") => DetailLevel::Full,
            Some("standard") => DetailLevel::Standard,
            _ => DetailLevel::Summary,
        };
        self.handle_lookup(&input.entity_id, detail_level)
            .await
            .map(Json)
            .map_err(ToolError::from)
    }

    #[tool(
        description = "Create a new character with optional profile (wound, secret, desire, contradiction categories)."
    )]
    #[instrument(name = "mcp.create_character", skip_all)]
    pub async fn create_character(
        &self,
        request: Parameters<CreateCharacterInput>,
    ) -> Result<Json<MutationResponse>, ToolError> {
        let Parameters(input) = request;
        self.handle_create_character(
            input.id,
            input.name,
            input.role,
            input.aliases,
            input.description,
            input.profile,
        )
        .await
        .map(Json)
        .map_err(ToolError::from)
    }

    #[tool(
        description = "Create a directional relationship between two characters (e.g., ally, enemy, mentor, rival)."
    )]
    #[instrument(name = "mcp.create_relationship", skip_all)]
    pub async fn create_relationship(
        &self,
        request: Parameters<CreateRelationshipInput>,
    ) -> Result<Json<MutationResponse>, ToolError> {
        let Parameters(input) = request;
        self.handle_create_relationship(
            input.from_character_id,
            input.to_character_id,
            input.rel_type,
            input.subtype,
            input.label,
        )
        .await
        .map(Json)
        .map_err(ToolError::from)
    }

    #[tool(
        description = "Dramatic irony report: knowledge asymmetries that create tension. Optionally focus on one character."
    )]
    #[instrument(name = "mcp.irony_report", skip_all)]
    pub async fn irony_report(
        &self,
        request: Parameters<IronyReportInput>,
    ) -> Result<Json<QueryResponse>, ToolError> {
        let Parameters(input) = request;
        self.handle_dramatic_irony_report(
            input.character_id,
            input.min_scene_threshold.unwrap_or(3),
        )
        .await
        .map(Json)
        .map_err(ToolError::from)
    }

    #[tool(
        description = "Update any entity's fields. Pass entity_id and a JSON object of fields to modify."
    )]
    #[instrument(name = "mcp.update_entity", skip_all)]
    pub async fn update_entity(
        &self,
        request: Parameters<UpdateEntityInput>,
    ) -> Result<Json<MutationResponse>, ToolError> {
        let Parameters(input) = request;
        self.handle_update(&input.entity_id, input.fields)
            .await
            .map(Json)
            .map_err(ToolError::from)
    }

    #[tool(
        description = "Compare what two characters know: what A knows that B doesn't, and vice versa. Reveals information asymmetry for dialogue and plot."
    )]
    #[instrument(name = "mcp.knowledge_asymmetries", skip_all)]
    pub async fn knowledge_asymmetries(
        &self,
        request: Parameters<KnowledgeAsymmetriesInput>,
    ) -> Result<Json<QueryResponse>, ToolError> {
        let Parameters(input) = request;
        self.handle_knowledge_asymmetries(&input.character_a, &input.character_b)
            .await
            .map(Json)
            .map_err(ToolError::from)
    }

    #[tool(
        description = "Validate an entity for consistency issues: fact violations, timeline problems, relationship conflicts."
    )]
    #[instrument(name = "mcp.validate_entity", skip_all)]
    pub async fn validate_entity(
        &self,
        request: Parameters<ValidateEntityInput>,
    ) -> Result<Json<QueryResponse>, ToolError> {
        let Parameters(input) = request;
        self.handle_validate_entity_query(&input.entity_id)
            .await
            .map(Json)
            .map_err(ToolError::from)
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

3-tier tool system for narrative world state management.

## Essential Tools (use first)
- semantic_search — Find by meaning/theme (concepts, feelings, themes)
- dossier — Full character analysis (network, knowledge, perceptions, arc)
- scene_prep — Scene preparation (pairwise dynamics, irony, tensions, facts)
- overview — World overview (entity counts, summaries)
- record_knowledge — Record what a character knows or believes

## Standard Tools
- search — Find by keyword (names, titles)
- lookup — Get entity by ID (full details)
- create_character — Create with profile (wound, secret, desire, contradiction)
- create_relationship — Link two characters
- irony_report — Knowledge asymmetries causing dramatic irony
- update_entity — Modify any entity's fields
- knowledge_asymmetries — What A knows that B doesn't, and vice versa
- validate_entity — Check consistency (fact violations, timeline, relationships)

## Advanced Tools (parameterized, 70 operations)
- query(operation) — 40 read ops: graph traversal, arc history/comparison/drift, perception gap/matrix/shift, centrality, influence, clustering, ...
- mutate(operation) — 25 write ops: batch create, import YAML, backfill embeddings, baseline arcs, protect entity, ...
- session(operation) — get_context, pin_entity, unpin_entity
- export_world — Export to YAML
- generate_graph — Mermaid diagram

## Resources
- narra://session/context — Hot entities, pinned items
- narra://world/overview — Current world state summary
- narra://entity/{type}:{id} — Full entity view
- narra://character/{id}/dossier — Character analysis
- narra://consistency/issues — Current violations
- narra://operations/guide — Categorized operation list with decision trees
- narra://schema/import-template — YAML import template

## Key Patterns
- Start sessions: session(get_context), then overview
- Before deleting: query(analyze_impact) first
- Consistency: CRITICAL violations block mutations
- After data entry: mutate(backfill_embeddings) once
- IDs: lowercase table:slug format (character:alice)
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
                    Annotated::new(
                        RawResource {
                            uri: "narra://world/overview".to_string(),
                            name: "World Overview".to_string(),
                            title: None,
                            description: Some(
                                "Current world state summary: entity counts and summaries by type"
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
                            uri: "narra://operations/guide".to_string(),
                            name: "Operations Guide".to_string(),
                            title: None,
                            description: Some(
                                "Categorized operation list with decision trees for finding the right tool"
                                    .to_string()
                            ),
                            mime_type: Some("text/markdown".to_string()),
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
        } else if uri == "narra://world/overview" {
            self.read_world_overview_resource(uri).await
        } else if uri == "narra://operations/guide" {
            self.read_operations_guide_resource(uri)
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
                Prompt::new(
                    "getting_started",
                    Some("First-time orientation: discover world state and learn the tool system"),
                    Some(vec![PromptArgument {
                        name: "focus".into(),
                        title: None,
                        description: Some(
                            "Focus area: general, characters, scenes, or consistency".into(),
                        ),
                        required: Some(false),
                    }]),
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
            "getting_started" => Ok(get_getting_started_prompt(request.arguments)),
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
            emotion_service: ctx.emotion_service.clone(),
            theme_service: ctx.theme_service.clone(),
            ner_service: ctx.ner_service.clone(),
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

    async fn read_world_overview_resource(
        &self,
        uri: &str,
    ) -> Result<ReadResourceResult, McpError> {
        let content = get_world_overview_resource(self)
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

    fn read_operations_guide_resource(&self, uri: &str) -> Result<ReadResourceResult, McpError> {
        Ok(ReadResourceResult {
            contents: vec![ResourceContents::TextResourceContents {
                uri: uri.to_string(),
                mime_type: Some("text/markdown".to_string()),
                text: get_operations_guide(),
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
            .handle_character_dossier(
                character_id,
                None,
                DEFAULT_TOKEN_BUDGET,
                crate::services::noop_progress(),
            )
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
    tracing::info!("MCP server listening on stdio (18 tools)");

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
