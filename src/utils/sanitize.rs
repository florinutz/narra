//! Entity ID validation for SurrealDB query safety.
//!
//! Entity IDs follow the `table:key` format (e.g., `character:alice`).
//! These functions validate the format to prevent SurrealQL injection
//! when building queries with `format!()`.

use crate::NarraError;

/// Allowed characters in the key portion of an entity ID.
/// Matches SurrealDB's record ID syntax: alphanumeric, underscores, hyphens.
fn is_valid_key_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// Known entity table names in the Narra schema.
const KNOWN_TABLES: &[&str] = &[
    "character",
    "location",
    "event",
    "scene",
    "fact",
    "note",
    "knows",
    "perceives",
    "relates_to",
    "annotation",
    "knowledge",
];

/// Validate that `entity_id` is a safe `table:key` format.
///
/// Returns the (table, key) parts if valid.
/// Returns `NarraError::Validation` if the format is invalid.
///
/// ```ignore
/// let (table, key) = validate_entity_id("character:alice")?;
/// assert_eq!(table, "character");
/// assert_eq!(key, "alice");
/// ```
pub fn validate_entity_id(entity_id: &str) -> Result<(&str, &str), NarraError> {
    let (table, key) = entity_id.split_once(':').ok_or_else(|| {
        NarraError::Validation(format!(
            "Invalid entity ID '{}': expected 'table:key' format",
            entity_id
        ))
    })?;

    if table.is_empty() || !table.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
        return Err(NarraError::Validation(format!(
            "Invalid entity ID '{}': table name must be lowercase alphanumeric with underscores",
            entity_id
        )));
    }

    if key.is_empty() || !key.chars().all(is_valid_key_char) {
        return Err(NarraError::Validation(format!(
            "Invalid entity ID '{}': key must be alphanumeric with underscores or hyphens",
            entity_id
        )));
    }

    Ok((table, key))
}

/// Validate that a bare key (without table prefix) is safe for query interpolation.
///
/// Used when the table part has already been stripped (e.g., after `id.split(':').nth(1)`).
pub fn validate_key(key: &str) -> Result<&str, NarraError> {
    if key.is_empty() || !key.chars().all(is_valid_key_char) {
        return Err(NarraError::Validation(format!(
            "Invalid key '{}': must be alphanumeric with underscores or hyphens",
            key
        )));
    }
    Ok(key)
}

/// Validate that a table name is known and safe.
pub fn validate_table(table: &str) -> Result<&str, NarraError> {
    if KNOWN_TABLES.contains(&table) {
        Ok(table)
    } else if !table.is_empty() && table.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
        // Unknown but syntactically valid — allow it
        Ok(table)
    } else {
        Err(NarraError::Validation(format!(
            "Invalid table name '{}'",
            table
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_entity_ids() {
        assert!(validate_entity_id("character:alice").is_ok());
        assert!(validate_entity_id("location:dark_manor").is_ok());
        assert!(validate_entity_id("event:battle-1").is_ok());
        assert!(validate_entity_id("scene:ch3_scene2").is_ok());
        assert!(validate_entity_id("annotation:abc123").is_ok());
    }

    #[test]
    fn test_invalid_entity_ids() {
        assert!(validate_entity_id("").is_err());
        assert!(validate_entity_id("nocolon").is_err());
        assert!(validate_entity_id(":nokey").is_err());
        assert!(validate_entity_id("character:").is_err());
        assert!(validate_entity_id("character:alice; DROP TABLE").is_err());
        assert!(validate_entity_id("character:alice\nDROP").is_err());
        assert!(validate_entity_id("Character:alice").is_err()); // uppercase table
        assert!(validate_entity_id("character:alice's").is_err()); // apostrophe
    }

    #[test]
    fn test_valid_keys() {
        assert!(validate_key("alice").is_ok());
        assert!(validate_key("dark_manor").is_ok());
        assert!(validate_key("battle-1").is_ok());
        assert!(validate_key("abc123").is_ok());
    }

    #[test]
    fn test_invalid_keys() {
        assert!(validate_key("").is_err());
        assert!(validate_key("alice; DROP").is_err());
        assert!(validate_key("al'ice").is_err());
    }

    #[test]
    fn test_validate_entity_id_parts() {
        let (table, key) = validate_entity_id("character:alice").unwrap();
        assert_eq!(table, "character");
        assert_eq!(key, "alice");
    }

    #[test]
    fn test_validate_table() {
        assert!(validate_table("character").is_ok());
        assert!(validate_table("relates_to").is_ok());
        assert!(validate_table("custom_table").is_ok()); // unknown but valid
        assert!(validate_table("").is_err());
        assert!(validate_table("Character").is_err()); // uppercase
    }

    // -- Property-based tests --

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_valid_ids_always_parse(
                table in "[a-z][a-z_]{0,10}",
                key in "[a-zA-Z0-9][a-zA-Z0-9_-]{0,20}",
            ) {
                let id = format!("{}:{}", table, key);
                let result = validate_entity_id(&id);
                prop_assert!(result.is_ok(), "Should parse valid ID: {}", id);
            }

            #[test]
            fn prop_injection_never_passes(
                table in "[a-z]{3,8}",
                payload in ".*(;|DROP|DELETE|UPDATE|INSERT|SELECT|--|').*",
            ) {
                let id = format!("{}:{}", table, payload);
                // If payload contains any non-alnum/underscore/hyphen chars, should fail
                if payload.chars().any(|c| !c.is_ascii_alphanumeric() && c != '_' && c != '-') {
                    prop_assert!(validate_entity_id(&id).is_err(),
                        "Injection payload should be rejected: {}", id);
                }
            }
        }
    }
}
