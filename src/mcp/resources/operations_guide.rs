//! Operations guide MCP resource.
//!
//! Static reference document categorizing all operations with decision trees
//! to help users find the right tool for their need.

/// Get the operations guide as a static text resource.
pub fn get_operations_guide() -> String {
    r#"# Narra Operations Guide

## Decision Tree: Finding the Right Tool

### "I want to find something..."
- By name/keyword → `search`
- By concept/theme → `semantic_search`
- By exact ID → `lookup`
- By meaning + name → `query(hybrid_search)`
- References to entity → `query(reverse_query)`

### "I want to understand a character..."
- Full analysis → `dossier`
- Voice/dialogue style → `query(character_voice)`
- Knowledge gaps → `query(knowledge_gap_analysis)`
- Relationship map → `query(relationship_strength_map)`
- How they've changed → `query(arc_history)`
- How others see them → `query(perception_matrix)`

### "I want to prepare a scene..."
- Scene dynamics → `scene_prep`
- Knowledge asymmetries → `knowledge_asymmetries`
- Dramatic irony → `irony_report`
- Tension between characters → `query(tension_matrix)`

### "I want to create something..."
- Character → `create_character`
- Relationship → `create_relationship`
- Location → `mutate(create_location)`
- Event → `mutate(create_event)`
- Scene → `mutate(create_scene)`
- Knowledge entry → `record_knowledge`
- Fact/rule → `mutate(create_fact)`
- Note → `mutate(create_note)`
- Many at once → `mutate(batch_create_*)`
- Import YAML → `mutate(import_yaml)`

### "I want to modify something..."
- Update fields → `update_entity`
- Delete entity → `mutate(delete)` (run `query(analyze_impact)` first)
- Protect entity → `mutate(protect_entity)`

### "I want to check consistency..."
- Single entity → `validate_entity`
- Deep investigation → `query(investigate_contradictions)`
- All issues → read `narra://consistency/issues`
- Impact preview → `query(analyze_impact)`

### "I want to analyze the narrative..."
- World situation → `query(situation_report)`
- Unresolved tensions → `query(unresolved_tensions)`
- Thematic gaps → `query(thematic_gaps)`
- Theme clusters → `query(thematic_clustering)`
- Narrative threads → `query(narrative_threads)`
- Knowledge conflicts → `query(knowledge_conflicts)`

### "I want to track character evolution..."
- Arc history → `query(arc_history)`
- Compare arcs → `query(arc_comparison)`
- Most changed → `query(arc_drift)`
- State at event → `query(arc_moment)`
- Growth direction → `query(growth_vector)`
- Convergence → `query(convergence_analysis)`

### "I want to explore perspectives..."
- How A sees B → `query(perception_gap)`
- All views of X → `query(perception_matrix)`
- View changes → `query(perception_shift)`
- Search perspectives → `query(perspective_search)`
- Misperception analysis → `query(misperception_vector)`

### "I want graph/network analysis..."
- Traverse graph → `query(graph_traversal)`
- Path between characters → `query(connection_path)`
- Centrality metrics → `query(centrality_metrics)`
- Influence propagation → `query(influence_propagation)`
- Similar relationships → `query(similar_relationships)`

### "I want what-if scenarios..."
- Knowledge what-if → `query(what_if)`
- Semantic midpoint → `query(semantic_midpoint)`

### "I want system operations..."
- Backfill embeddings → `mutate(backfill_embeddings)`
- Baseline arc snapshots → `mutate(baseline_arc_snapshots)`
- Embedding health → `query(embedding_health)`
- Export world → `export_world`
- Generate graph → `generate_graph`
- Session context → `session(get_context)`
- Pin/unpin → `session(pin_entity)` / `session(unpin_entity)`

## Tool Count Summary
- Essential dedicated tools: 5
- Standard dedicated tools: 8
- Parameterized query operations: 40
- Parameterized mutate operations: 25
- Session operations: 3
- Utility tools: 2 (export_world, generate_graph)
- Total: 18 tools covering 83 operations
"#
    .to_string()
}
