//! Helper functions to convert strongly-typed request enums to free-form MCP input wrappers.
//!
//! The MCP server uses free-form input types (QueryInput, MutationInput, SessionInput) for
//! better MCP tool schema compatibility. These helpers convert from the strongly-typed
//! request enums (QueryRequest, MutationRequest, SessionRequest) used in tests.

use narra::mcp::{
    MutationInput, MutationRequest, QueryInput, QueryRequest, SessionInput, SessionRequest,
};

/// Convert QueryRequest to QueryInput for MCP tool testing.
pub fn to_query_input(request: QueryRequest) -> QueryInput {
    // Serialize to JSON value
    let value = serde_json::to_value(&request).expect("Failed to serialize QueryRequest");

    // Extract operation and params
    let mut obj = value.as_object().expect("Request should be object").clone();
    let operation = obj
        .remove("operation")
        .and_then(|v| v.as_str().map(String::from))
        .expect("Request should have operation field");

    QueryInput {
        operation,
        token_budget: None,
        params: obj,
    }
}

/// Convert MutationRequest to MutationInput for MCP tool testing.
pub fn to_mutation_input(request: MutationRequest) -> MutationInput {
    // Serialize to JSON value
    let value = serde_json::to_value(&request).expect("Failed to serialize MutationRequest");

    // Extract operation and params
    let mut obj = value.as_object().expect("Request should be object").clone();
    let operation = obj
        .remove("operation")
        .and_then(|v| v.as_str().map(String::from))
        .expect("Request should have operation field");

    MutationInput {
        operation,
        params: obj,
    }
}

/// Convert SessionRequest to SessionInput for MCP tool testing.
pub fn to_session_input(request: SessionRequest) -> SessionInput {
    // Serialize to JSON value
    let value = serde_json::to_value(&request).expect("Failed to serialize SessionRequest");

    // Extract operation and params
    let mut obj = value.as_object().expect("Request should be object").clone();
    let operation = obj
        .remove("operation")
        .and_then(|v| v.as_str().map(String::from))
        .expect("Request should have operation field");

    SessionInput {
        operation,
        params: obj,
    }
}
