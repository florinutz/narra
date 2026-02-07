//! Dramatic irony prompt for knowledge asymmetry analysis.
//!
//! This prompt guides Claude through analyzing knowledge differences between
//! characters to identify opportunities for dramatic tension.

use rmcp::model::{GetPromptResult, PromptMessage, PromptMessageRole};
use serde_json::{Map, Value};

/// Get the dramatic irony prompt with optional arguments.
///
/// # Arguments
/// - `character_a` (required): First character ID to compare (e.g., "character:alice")
/// - `character_b` (optional): Second character ID (if omitted, compare against all characters)
/// - `topic` (optional): Focus on knowledge about specific entity/topic
pub fn get_dramatic_irony_prompt(args: Option<Map<String, Value>>) -> GetPromptResult {
    let character_a = args
        .as_ref()
        .and_then(|a| a.get("character_a"))
        .and_then(|v| v.as_str())
        .unwrap_or("(no character specified)");

    let character_b = args
        .as_ref()
        .and_then(|a| a.get("character_b"))
        .and_then(|v| v.as_str());

    let topic = args
        .as_ref()
        .and_then(|a| a.get("topic"))
        .and_then(|v| v.as_str());

    let comparison_target = match character_b {
        Some(b) => b.to_string(),
        None => "all other characters".to_string(),
    };

    let topic_focus = match topic {
        Some(t) => format!(" focusing on knowledge about '{}'", t),
        None => String::new(),
    };

    GetPromptResult {
        description: Some(format!(
            "Analyze knowledge asymmetry between {} and {}{}",
            character_a, comparison_target, topic_focus
        )),
        messages: vec![
            // Assistant explains dramatic irony concept
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                format!(
                    r#"I'll analyze the knowledge asymmetry between {character_a} and {comparison_target}{topic_focus}.

**What is Dramatic Irony?**
Dramatic irony occurs when the audience (or some characters) know something that other characters don't. This creates tension, humor, or tragedy depending on the context.

**Analysis Approach:**
1. Map what each character knows
2. Identify knowledge gaps (what A knows that B doesn't, and vice versa)
3. Find tension opportunities where asymmetry could drive conflict or revelation
4. Suggest scenes that leverage these knowledge differences"#,
                    character_a = character_a,
                    comparison_target = comparison_target,
                    topic_focus = topic_focus
                ),
            ),
            // User request template
            PromptMessage::new_text(
                PromptMessageRole::User,
                format!(
                    "Analyze knowledge asymmetry between {} and {}{}.",
                    character_a, comparison_target, topic_focus
                ),
            ),
            // Assistant workflow steps
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                format!(
                    r#"I'll run the dramatic irony analysis workflow:

**Step 1: Gather Knowledge States**
Use the `query` tool with temporal operations to get each character's knowledge:
```json
{{"operation": "temporal", "query_type": "knowledge_at_time", "entity_id": "{character_a}", "time_point": "current"}}
```
{character_b_query}

**Step 2: Compare Knowledge Sets**
For each piece of knowledge, identify:
- **A knows, B doesn't**: Secrets A holds over B
- **B knows, A doesn't**: Blind spots A has
- **Both know differently**: Contradicting beliefs (most potent for irony)

**Step 3: Identify Tension Opportunities**

| Pattern | Example | Tension Type |
|---------|---------|--------------|
| Secret knowledge | A knows B's true parentage | Suspense |
| Misconception | A thinks B loves them (B doesn't) | Tragedy/Comedy |
| Information gap | A doesn't know about the danger | Suspense |
| Contradicting beliefs | A thinks X is dead, B knows X is alive | Dramatic reveal |

**Step 4: Suggest Scene Opportunities**
For each significant asymmetry, suggest:
- A scene where the asymmetry creates tension
- How a reveal could play out
- Whether to maintain or resolve the irony

**Step 5: Note Consistency Issues**
If I find knowledge that contradicts universe facts (unintentional), I'll flag it.
Use the `check_consistency` prompt if deeper validation is needed.

{topic_note}"#,
                    character_a = character_a,
                    character_b_query = match character_b {
                        Some(b) => format!(
                            "```json\n{{\"operation\": \"temporal\", \"query_type\": \"knowledge_at_time\", \"entity_id\": \"{}\", \"time_point\": \"current\"}}\n```",
                            b
                        ),
                        None => "Repeat for all characters connected to the target.".to_string(),
                    },
                    topic_note = match topic {
                        Some(t) => format!("\n*Focusing analysis on knowledge related to: {}*", t),
                        None => String::new(),
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
    fn test_irony_prompt_minimal_args() {
        let mut args = Map::new();
        args.insert(
            "character_a".to_string(),
            Value::String("character:alice".to_string()),
        );

        let result = get_dramatic_irony_prompt(Some(args));
        assert!(result.description.is_some());
        assert_eq!(result.messages.len(), 3);
        assert!(result.description.unwrap().contains("character:alice"));
    }

    #[test]
    fn test_irony_prompt_with_character_b() {
        let mut args = Map::new();
        args.insert(
            "character_a".to_string(),
            Value::String("character:alice".to_string()),
        );
        args.insert(
            "character_b".to_string(),
            Value::String("character:bob".to_string()),
        );

        let result = get_dramatic_irony_prompt(Some(args));
        let desc = result.description.unwrap();
        assert!(desc.contains("alice"));
        assert!(desc.contains("bob"));
    }

    #[test]
    fn test_irony_prompt_with_topic() {
        let mut args = Map::new();
        args.insert(
            "character_a".to_string(),
            Value::String("character:alice".to_string()),
        );
        args.insert("topic".to_string(), Value::String("the murder".to_string()));

        let result = get_dramatic_irony_prompt(Some(args));
        assert!(result.description.unwrap().contains("the murder"));
    }

    #[test]
    fn test_irony_prompt_no_args() {
        let result = get_dramatic_irony_prompt(None);
        assert!(result.description.is_some());
        // Should handle missing required argument gracefully
        assert!(result
            .description
            .unwrap()
            .contains("no character specified"));
    }
}
