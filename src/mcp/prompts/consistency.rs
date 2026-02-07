//! Consistency check prompt for validating entities.
//!
//! This prompt guides Claude through a structured workflow for validating
//! entity consistency and suggesting fixes for any violations found.

use rmcp::model::{GetPromptResult, PromptMessage, PromptMessageRole};
use serde_json::{Map, Value};

/// Get the consistency check prompt with optional arguments.
///
/// # Arguments
/// - `entity_id`: Focus validation on specific entity (e.g., "character:alice"), or check all if omitted
/// - `severity_filter`: Minimum severity to show ("critical", "warning", or "info"). Default: all
pub fn get_consistency_check_prompt(args: Option<Map<String, Value>>) -> GetPromptResult {
    let entity_id = args
        .as_ref()
        .and_then(|a| a.get("entity_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("all entities");

    let severity_filter = args
        .as_ref()
        .and_then(|a| a.get("severity_filter"))
        .and_then(|v| v.as_str());

    let severity_instruction = match severity_filter {
        Some("critical") => "Focus only on CRITICAL severity issues.",
        Some("warning") => "Show WARNING and CRITICAL severity issues.",
        _ => "Show all issues (CRITICAL, WARNING, and INFO).",
    };

    let target_description = if entity_id == "all entities" {
        "all entities in the world".to_string()
    } else {
        format!("the entity '{}'", entity_id)
    };

    GetPromptResult {
        description: Some(format!(
            "Validate {} for consistency issues and suggest fixes",
            target_description
        )),
        messages: vec![
            // Assistant introduces the workflow
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                format!(
                    "I'll help you validate {} for consistency issues. This workflow will:\n\n\
                    1. Check for fact violations (contradictions with universe rules)\n\
                    2. Check for timeline contradictions (knowledge before learning, impossible sequences)\n\
                    3. Check for relationship conflicts (circular hierarchies, asymmetric perceptions)\n\n\
                    {}",
                    target_description,
                    severity_instruction
                ),
            ),
            // User request template
            PromptMessage::new_text(
                PromptMessageRole::User,
                format!("Validate {} for consistency issues.", entity_id),
            ),
            // Assistant workflow steps
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                format!(
                    r#"I'll now run the consistency validation workflow:

**Step 1: Run Validation**
Use the `validate` tool with the appropriate operation:
- For specific entity: `{{"operation": "validate_entity", "entity_id": "{entity_id}"}}`
- For investigation: `{{"operation": "investigate_contradictions", "entity_id": "{entity_id}", "depth": 3}}`

**Step 2: Group Issues by Severity**
Present results grouped by severity (most critical first):
1. **CRITICAL** - Must be fixed immediately (blocks mutations)
2. **WARNING** - Should be addressed (allows mutations with notice)
3. **INFO** - Informational (potential intentional contradictions)

**Step 3: For Each Issue, Show**
- The violation message
- Suggested fix (if available)
- Confidence level
- Related entities

**Step 4: If No Issues Found**
- Confirm the entity/world is consistent
- Suggest related checks (e.g., "You might also want to validate related characters")

**Step 5: Offer Resolution**
For multiple related issues, offer to fix them in batch:
- "I found 3 timeline violations. Would you like me to suggest a timeline reorganization?"
- "There are 2 asymmetric relationships. Should I help reconcile them?"

{severity_note}"#,
                    entity_id = entity_id,
                    severity_note = if severity_filter.is_some() {
                        format!("\n*Filtering to {} severity and above.*", severity_filter.unwrap_or("all"))
                    } else {
                        String::new()
                    }
                ),
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistency_prompt_no_args() {
        let result = get_consistency_check_prompt(None);
        assert!(result.description.is_some());
        assert_eq!(result.messages.len(), 3);
        assert!(result.description.unwrap().contains("all entities"));
    }

    #[test]
    fn test_consistency_prompt_with_entity_id() {
        let mut args = Map::new();
        args.insert(
            "entity_id".to_string(),
            Value::String("character:alice".to_string()),
        );

        let result = get_consistency_check_prompt(Some(args));
        assert!(result.description.unwrap().contains("character:alice"));
    }

    #[test]
    fn test_consistency_prompt_with_severity_filter() {
        let mut args = Map::new();
        args.insert(
            "severity_filter".to_string(),
            Value::String("critical".to_string()),
        );

        let result = get_consistency_check_prompt(Some(args));
        // Check that the workflow mentions filtering
        let workflow_msg = &result.messages[2];
        if let rmcp::model::PromptMessageContent::Text { text } = &workflow_msg.content {
            assert!(text.contains("critical"));
        }
    }
}
