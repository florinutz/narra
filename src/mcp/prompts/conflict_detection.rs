//! Conflict detection prompt for identifying active and latent conflicts.
//!
//! Guides the LLM through discovering conflicts from tensions, knowledge
//! asymmetries, and perception gaps, with escalation and resolution paths.

use rmcp::model::{GetPromptResult, PromptMessage, PromptMessageRole};
use serde_json::{Map, Value};

/// Get the conflict detection prompt.
///
/// # Arguments
/// - `scope` (optional): "all" for world-wide scan, or a specific entity ID to focus on
pub fn get_conflict_detection_prompt(args: Option<Map<String, Value>>) -> GetPromptResult {
    let scope = args
        .as_ref()
        .and_then(|a| a.get("scope"))
        .and_then(|v| v.as_str());

    let scope_display = match scope {
        Some("all") | None => "the entire narrative world".to_string(),
        Some(id) => format!("conflicts involving {}", id),
    };

    let scope_query = match scope {
        Some("all") | None => String::new(),
        Some(id) => format!(", \"character_id\": \"{}\"", id),
    };

    GetPromptResult {
        description: Some(format!("Detect conflicts in {}", scope_display)),
        messages: vec![
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                format!(
                    r#"I'll systematically identify active and latent conflicts in {scope_display}.

**Conflict Types I'll Look For:**
- **Active conflicts**: Characters with high tension and opposing knowledge
- **Latent conflicts**: Hidden tensions that could erupt (misperceptions, secrets)
- **Structural conflicts**: Role-based oppositions (protagonist/antagonist dynamics)
- **Knowledge conflicts**: Characters who believe contradictory things"#,
                    scope_display = scope_display,
                ),
            ),
            PromptMessage::new_text(
                PromptMessageRole::User,
                format!("Find all conflicts in {}.", scope_display),
            ),
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                format!(
                    r#"I'll run the conflict detection workflow:

**Step 1: Gather Tension Data**
```json
{{"operation": "tension_matrix", "min_tension": 1{scope_query}}}
```
This reveals all character pairs with elevated tension.

**Step 2: Knowledge Conflicts**
```json
{{"operation": "knowledge_conflicts"{scope_query}}}
```
Characters who believe contradictory things create natural conflict vectors.

**Step 3: Dramatic Irony Scan**
```json
{{"operation": "dramatic_irony_report"}}
```
Knowledge asymmetries reveal latent conflicts — what happens when secrets come out?

**Step 4: Situation Overview**
```json
{{"operation": "situation_report"}}
```
The big picture: irony highlights, high-tension pairs, and narrative suggestions.

**Step 5: Classify and Prioritize**

For each conflict found, I'll categorize:

| Priority | Conflict Type | Parties | Status | Escalation Risk |
|----------|--------------|---------|--------|-----------------|
| P1 | Active + High Tension | Named characters | Active/Latent | High/Medium/Low |

**Step 6: Analyze Each Conflict**

For each significant conflict:
- **Root cause**: What drives this conflict?
- **Current state**: Active, simmering, or dormant?
- **Escalation path**: What would make it worse?
- **Resolution path**: What would resolve it?
- **Narrative value**: How does this conflict serve the story?
- **Dependencies**: Other conflicts that feed into or from this one

**Step 7: Conflict Map**
I'll produce a visual summary showing how conflicts interconnect — often one character's resolution creates another's escalation.

Let me start gathering the data."#,
                    scope_query = scope_query,
                ),
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflict_detection_all() {
        let mut args = Map::new();
        args.insert("scope".to_string(), Value::String("all".to_string()));

        let result = get_conflict_detection_prompt(Some(args));
        assert!(result.description.is_some());
        assert_eq!(result.messages.len(), 3);
        assert!(result
            .description
            .unwrap()
            .contains("entire narrative world"));
    }

    #[test]
    fn test_conflict_detection_scoped() {
        let mut args = Map::new();
        args.insert(
            "scope".to_string(),
            Value::String("character:alice".to_string()),
        );

        let result = get_conflict_detection_prompt(Some(args));
        let desc = result.description.unwrap();
        assert!(desc.contains("alice"));
    }

    #[test]
    fn test_conflict_detection_no_args() {
        let result = get_conflict_detection_prompt(None);
        assert!(result
            .description
            .unwrap()
            .contains("entire narrative world"));
    }
}
