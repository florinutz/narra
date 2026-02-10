//! World consistency oracle prompt for validating proposed plot actions.
//!
//! Guides the LLM through checking a proposed action against relevant facts,
//! affected entities, potential contradictions, and timeline concerns.

use rmcp::model::{GetPromptResult, PromptMessage, PromptMessageRole};
use serde_json::{Map, Value};

/// Get the world consistency oracle prompt.
///
/// # Arguments
/// - `proposed_action` (required): Natural language description of the proposed plot action
/// - `affected_entities` (optional): Comma-separated entity IDs that might be affected
pub fn get_consistency_oracle_prompt(args: Option<Map<String, Value>>) -> GetPromptResult {
    let proposed_action = args
        .as_ref()
        .and_then(|a| a.get("proposed_action"))
        .and_then(|v| v.as_str())
        .unwrap_or("(no action specified)");

    let affected_entities = args
        .as_ref()
        .and_then(|a| a.get("affected_entities"))
        .and_then(|v| v.as_str());

    let entities_note = match affected_entities {
        Some(e) => format!("\nPotentially affected entities: {}", e),
        None => String::new(),
    };

    let entities_lookup = match affected_entities {
        Some(entities) => {
            let ids: Vec<&str> = entities.split(',').map(|s| s.trim()).collect();
            ids.iter()
                .map(|id| {
                    format!(
                        "```json\n{{\"operation\": \"lookup\", \"entity_id\": \"{}\"}}\n```",
                        id
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        None => {
            "Use `search` or `hybrid_search` to find entities mentioned in the proposed action."
                .to_string()
        }
    };

    GetPromptResult {
        description: Some(format!(
            "Check consistency of: {}",
            if proposed_action.len() > 60 {
                format!("{}...", &proposed_action[..60])
            } else {
                proposed_action.to_string()
            }
        )),
        messages: vec![
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                format!(
                    r#"I'll check whether this proposed action is consistent with the established world:

**Proposed Action:** {proposed_action}{entities_note}

**Validation Dimensions:**
1. Universe facts — does this violate any established rules?
2. Timeline consistency — does this fit the event sequence?
3. Knowledge plausibility — can characters know what they'd need to know?
4. Relationship coherence — does this match established dynamics?
5. Spatial consistency — are characters where they need to be?"#,
                    proposed_action = proposed_action,
                    entities_note = entities_note,
                ),
            ),
            PromptMessage::new_text(
                PromptMessageRole::User,
                format!(
                    "Check if this action is consistent with the world: {}",
                    proposed_action
                ),
            ),
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                format!(
                    r#"I'll run the consistency oracle workflow:

**Step 1: Identify Affected Entities**
{entities_lookup}

**Step 2: Check Universe Facts**
```json
{{"operation": "list_facts", "category": "all"}}
```
I'll cross-reference the proposed action against all established facts, especially those with "hard" enforcement.

**Step 3: Validate Affected Entities**
For each entity involved:
```json
{{"operation": "validate_entity", "entity_id": "<entity_id>"}}
```
This checks for existing inconsistencies that the action might worsen.

**Step 4: Impact Analysis**
For each primary entity affected:
```json
{{"operation": "analyze_impact", "entity_id": "<entity_id>", "proposed_change": "{proposed_action_escaped}"}}
```
This shows the ripple effects across the world.

**Step 5: Timeline Check**
Look at temporal knowledge and event sequences to verify the action fits chronologically.

**Step 6: Verdict**

I'll provide a structured assessment:

### Consistency Verdict
| Dimension | Status | Details |
|-----------|--------|---------|
| Universe Facts | PASS/WARN/FAIL | Any violated facts |
| Timeline | PASS/WARN/FAIL | Sequence issues |
| Knowledge | PASS/WARN/FAIL | Plausibility of character knowledge |
| Relationships | PASS/WARN/FAIL | Dynamic coherence |

### Issues Found
For each issue:
- **Severity**: Critical (blocks action), Warning (notable), Info (minor)
- **What**: The specific inconsistency
- **Why**: How the action creates it
- **Fix**: How to modify the action to avoid the issue

### Recommended Modifications
If issues exist, I'll suggest the minimal changes to make the action consistent.

### Side Effects
Even if consistent, I'll note downstream consequences:
- Knowledge states that need updating
- Relationships that would shift
- Future scenes that become possible or impossible

Let me start checking."#,
                    entities_lookup = entities_lookup,
                    proposed_action_escaped = proposed_action.replace('"', "\\\""),
                ),
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistency_oracle_with_action() {
        let mut args = Map::new();
        args.insert(
            "proposed_action".to_string(),
            Value::String("Alice discovers Bob's true identity at the tavern".to_string()),
        );

        let result = get_consistency_oracle_prompt(Some(args));
        assert!(result.description.is_some());
        assert_eq!(result.messages.len(), 3);
        assert!(result.description.unwrap().contains("Alice discovers"));
    }

    #[test]
    fn test_consistency_oracle_with_entities() {
        let mut args = Map::new();
        args.insert(
            "proposed_action".to_string(),
            Value::String("Alice kills the dragon".to_string()),
        );
        args.insert(
            "affected_entities".to_string(),
            Value::String("character:alice, event:dragon_fight".to_string()),
        );

        let result = get_consistency_oracle_prompt(Some(args));
        assert_eq!(result.messages.len(), 3);
        assert!(result.description.unwrap().contains("Alice kills"));
    }

    #[test]
    fn test_consistency_oracle_no_args() {
        let result = get_consistency_oracle_prompt(None);
        assert!(result.description.unwrap().contains("no action specified"));
    }

    #[test]
    fn test_consistency_oracle_long_action_truncated_in_description() {
        let mut args = Map::new();
        args.insert(
            "proposed_action".to_string(),
            Value::String(
                "Alice discovers that Bob has been secretly working with the antagonist to undermine the kingdom's defenses for the past three years"
                    .to_string(),
            ),
        );

        let result = get_consistency_oracle_prompt(Some(args));
        let desc = result.description.unwrap();
        assert!(desc.ends_with("..."));
    }
}
