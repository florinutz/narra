//! Getting started prompt for first-time orientation.
//!
//! Guides new users through discovering the world state and understanding
//! the tool system.

use rmcp::model::{GetPromptResult, PromptMessage, PromptMessageRole};
use serde_json::{Map, Value};

/// Get the getting started prompt.
///
/// Returns a multi-step workflow for orienting in a narrative world.
pub fn get_getting_started_prompt(args: Option<Map<String, Value>>) -> GetPromptResult {
    let focus = args
        .as_ref()
        .and_then(|a| a.get("focus"))
        .and_then(|v| v.as_str())
        .unwrap_or("general");

    let focus_instruction = match focus {
        "characters" => "\nFocus on character exploration: use `dossier` for deep dives and `knowledge_asymmetries` to find dramatic tension.",
        "scenes" => "\nFocus on scene preparation: use `scene_prep` with character IDs and `irony_report` for tension opportunities.",
        "consistency" => "\nFocus on validation: use `validate_entity` on key entities and read `narra://consistency/issues` for a full report.",
        _ => "",
    };

    GetPromptResult {
        description: Some("First-time orientation: discover world state and learn the tool system".to_string()),
        messages: vec![
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                format!(
                    "I'll help you get oriented with this narrative world. Here's the workflow:\n\n\
                    1. **Get context** — Check what's been happening recently\n\
                    2. **Survey the world** — See what entities exist\n\
                    3. **Explore key characters** — Deep-dive into protagonists\n\
                    4. **Check consistency** — Ensure the world is internally consistent\n\
                    {}",
                    focus_instruction
                ),
            ),
            PromptMessage::new_text(
                PromptMessageRole::User,
                "Help me get started with this narrative world.".to_string(),
            ),
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                r#"Here's the orientation workflow:

**Step 1: Session Context**
Use `session` with operation `get_context` to see recent activity, hot entities, and pinned items.

**Step 2: World Overview**
Use `overview` to get entity counts and summaries across all types (characters, locations, events, scenes).

**Step 3: Explore Characters**
For each protagonist, use `dossier` to get their full analysis: network position, knowledge, perceptions, and arc trajectory.

**Step 4: Check Dynamics**
Use `irony_report` to see knowledge asymmetries creating dramatic tension across the world.

**Step 5: Validate**
Use `validate_entity` on key entities to check for consistency issues (fact violations, timeline problems).

**Step 6: Prepare for Writing**
When ready to write a scene, use `scene_prep` with the participating character IDs to get pairwise dynamics, irony opportunities, and applicable facts.

**Tip**: For the full list of all 83 operations organized by use case, read the `narra://operations/guide` resource."#
                    .to_string(),
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_getting_started_prompt_no_args() {
        let result = get_getting_started_prompt(None);
        assert!(result.description.is_some());
        assert_eq!(result.messages.len(), 3);
    }

    #[test]
    fn test_getting_started_prompt_with_focus() {
        let mut args = Map::new();
        args.insert(
            "focus".to_string(),
            Value::String("characters".to_string()),
        );

        let result = get_getting_started_prompt(Some(args));
        let intro_msg = &result.messages[0];
        if let rmcp::model::PromptMessageContent::Text { text } = &intro_msg.content {
            assert!(text.contains("dossier"));
        }
    }
}
