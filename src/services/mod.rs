pub mod arc;
pub mod clustering;
pub mod composite;

pub mod consistency;
pub mod context;
pub mod emotion;
pub mod export;
pub mod graph;
pub mod graph_analytics;
pub mod impact;
pub mod import;
pub mod influence;
pub mod irony;
pub mod perception;
pub mod search;
pub mod summary;
pub mod temporal;
pub mod vector_ops;

pub use clustering::{ClusteringResult, ClusteringService, ThemeCluster};
pub use composite::{
    CharacterDossier, CompositeIntelligenceService, NarrativeMomentum, ScenePlan, SituationReport,
};
pub use consistency::{
    generate_suggested_fix, ConsistencyChecker, ConsistencyService, ConsistencySeverity,
    ValidationResult, Violation,
};
pub use context::{
    CachedContextService, ContextConfig, ContextResponse, ContextService, ScoredEntity,
};
pub use graph::{GraphOptions, GraphScope, GraphService, MermaidGraphService};
pub use graph_analytics::{CentralityMetric, CentralityResult, GraphAnalyticsService};
pub use impact::{
    AffectedEntity, Decision, DeferredImplication, ImpactAnalysis, ImpactAnalyzer, ImpactService,
    Severity,
};
pub use influence::{InfluencePath, InfluenceService, InfluenceStep, PropagationResult};
pub use irony::{IronyReport, IronyService, KnowledgeAsymmetry};
pub use search::{
    apply_rrf, EntityType, FilterOp, MetadataFilter, SearchFilter, SearchResult, SearchService,
    SurrealSearchService,
};
pub use summary::{
    CachedSummaryService, DetailLevel, EntityFullContent, EntitySummary, SummaryConfig,
    SummaryService,
};

pub use arc::{ArcComparisonResult, ArcHistoryResult, ArcMomentResult, ArcService};
pub use emotion::{EmotionService, LocalEmotionService, NoopEmotionService};
pub use perception::{
    PerceptionGapResult, PerceptionMatrixResult, PerceptionService, PerceptionShiftResult,
};
pub use temporal::{
    NarrativeNeighbor, NarrativeNeighborhood, NarrativePhase, PhaseDetectionResult, PhaseMember,
    PhaseWeights, TemporalService,
};
pub use vector_ops::{
    ConvergenceResult, GrowthVectorResult, MidpointResult, MisperceptionResult, VectorOpsService,
};
