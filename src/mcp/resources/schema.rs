use crate::mcp::types::NarraImport;

/// Get the hand-written YAML import template with inline comments and examples.
pub fn get_import_template() -> String {
    include_str!("import_template.yaml").to_string()
}

/// Get the auto-generated JSON Schema for the NarraImport type.
pub fn get_import_schema() -> String {
    let schema = schemars::schema_for!(NarraImport);
    serde_json::to_string_pretty(&schema)
        .unwrap_or_else(|e| format!("{{\"error\": \"Failed to serialize schema: {}\"}}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_parses_as_valid_import() {
        let template = get_import_template();
        let parsed: Result<NarraImport, _> = serde_yaml_ng::from_str(&template);
        assert!(
            parsed.is_ok(),
            "Template failed to parse: {:?}",
            parsed.err()
        );
        let import = parsed.unwrap();
        assert!(!import.characters.is_empty());
        assert!(!import.locations.is_empty());
        assert!(!import.events.is_empty());
        assert!(!import.scenes.is_empty());
    }

    #[test]
    fn schema_is_valid_json() {
        let schema = get_import_schema();
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&schema);
        assert!(
            parsed.is_ok(),
            "Schema is not valid JSON: {:?}",
            parsed.err()
        );
        let value = parsed.unwrap();
        assert!(value.get("$schema").is_some() || value.get("type").is_some());
    }

    #[test]
    #[ignore] // Run with: cargo test --lib print_import_schema -- --ignored --nocapture
    fn print_import_schema() {
        println!("{}", get_import_schema());
    }
}
