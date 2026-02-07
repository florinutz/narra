//! Unit tests for RRF (Reciprocal Rank Fusion) merge algorithm.
//!
//! Tests `apply_rrf` from `narra::services::search` — no database needed.

use narra::services::{apply_rrf, SearchResult};

fn make_result(id: &str, name: &str, score: f32) -> SearchResult {
    SearchResult {
        id: id.to_string(),
        entity_type: "character".to_string(),
        name: name.to_string(),
        score,
    }
}

#[test]
fn test_rrf_overlapping_results_merged_order() {
    // Both lists contain "alice" and "bob" — overlapping items should get higher RRF scores
    let keyword = vec![
        make_result("character:alice", "Alice", 10.0),
        make_result("character:bob", "Bob", 8.0),
        make_result("character:carol", "Carol", 5.0),
    ];
    let semantic = vec![
        make_result("character:bob", "Bob", 0.95),
        make_result("character:alice", "Alice", 0.90),
        make_result("character:dave", "Dave", 0.80),
    ];

    let merged = apply_rrf(&keyword, &semantic);

    // Both alice and bob appear in both lists, so they should rank highest
    assert_eq!(merged.len(), 4, "Should include all unique items");

    // alice: rank 1 in keyword (1/(60+1)) + rank 2 in semantic (1/(60+2))
    // bob: rank 2 in keyword (1/(60+2)) + rank 1 in semantic (1/(60+1))
    // Both have the same total RRF score, so tie-break by id ascending
    assert_eq!(merged[0].id, "character:alice");
    assert_eq!(merged[1].id, "character:bob");

    // carol and dave only appear in one list each
    assert!(merged[2..].iter().any(|r| r.id == "character:carol"));
    assert!(merged[2..].iter().any(|r| r.id == "character:dave"));
}

#[test]
fn test_rrf_disjoint_results_include_all() {
    let keyword = vec![
        make_result("character:alice", "Alice", 10.0),
        make_result("character:bob", "Bob", 8.0),
    ];
    let semantic = vec![
        make_result("character:carol", "Carol", 0.95),
        make_result("character:dave", "Dave", 0.90),
    ];

    let merged = apply_rrf(&keyword, &semantic);

    assert_eq!(merged.len(), 4, "All items from both lists should appear");
    let ids: Vec<&str> = merged.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"character:alice"));
    assert!(ids.contains(&"character:bob"));
    assert!(ids.contains(&"character:carol"));
    assert!(ids.contains(&"character:dave"));
}

#[test]
fn test_rrf_empty_inputs() {
    let empty: Vec<SearchResult> = vec![];

    // Both empty
    let merged = apply_rrf(&empty, &empty);
    assert!(merged.is_empty());

    // One empty, one has results
    let keyword = vec![make_result("character:alice", "Alice", 10.0)];
    let merged = apply_rrf(&keyword, &empty);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].id, "character:alice");

    let merged = apply_rrf(&empty, &keyword);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].id, "character:alice");
}

#[test]
fn test_rrf_single_source_preserves_order() {
    let keyword = vec![
        make_result("character:alice", "Alice", 10.0),
        make_result("character:bob", "Bob", 8.0),
        make_result("character:carol", "Carol", 5.0),
    ];
    let empty: Vec<SearchResult> = vec![];

    let merged = apply_rrf(&keyword, &empty);

    assert_eq!(merged.len(), 3);
    // RRF scores are 1/(60+1), 1/(60+2), 1/(60+3) — monotonically decreasing
    assert_eq!(merged[0].id, "character:alice");
    assert_eq!(merged[1].id, "character:bob");
    assert_eq!(merged[2].id, "character:carol");
}
