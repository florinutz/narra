//! Scene planning prompt for guided scene construction.
//!
//! Gathers pairwise dynamics, applicable facts, irony opportunities;
//! guides the LLM through scene construction with narrative awareness.

use rmcp::model::{GetPromptResult, PromptMessage, PromptMessageRole};
use serde_json::{Map, Value};

/// Get the scene planning prompt.
///
/// # Arguments
/// - `character_ids` (required): Comma-separated character IDs (e.g., "character:alice,character:bob")
/// - `location_id` (optional): Location for the scene
/// - `tone` (optional): Desired tone (e.g., "tense", "comedic", "intimate")
pub fn get_scene_planning_prompt(args: Option<Map<String, Value>>) -> GetPromptResult {
    let character_ids = args
        .as_ref()
        .and_then(|a| a.get("character_ids"))
        .and_then(|v| v.as_str())
        .unwrap_or("(no characters specified)");

    let location_id = args
        .as_ref()
        .and_then(|a| a.get("location_id"))
        .and_then(|v| v.as_str());

    let tone = args
        .as_ref()
        .and_then(|a| a.get("tone"))
        .and_then(|v| v.as_str());

    let char_list: Vec<&str> = character_ids.split(',').map(|s| s.trim()).collect();
    let char_display = char_list.join(", ");

    let location_note = match location_id {
        Some(loc) => format!(" at {}", loc),
        None => String::new(),
    };

    let tone_note = match tone {
        Some(t) => format!(" with a {} tone", t),
        None => String::new(),
    };

    let char_ids_json: Vec<String> = char_list.iter().map(|c| format!("\"{}\"", c)).collect();
    let char_ids_array = char_ids_json.join(", ");

    GetPromptResult {
        description: Some(format!(
            "Plan a scene with {}{}{}",
            char_display, location_note, tone_note
        )),
        messages: vec![
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                format!(
                    r#"I'll help you plan a scene with {char_display}{location_note}{tone_note}.

**Scene Planning Workflow:**
1. Gather character dynamics and relationships
2. Identify knowledge asymmetries and irony opportunities
3. Check applicable universe facts and constraints
4. Design scene beats that leverage tensions
5. Suggest dialogue approaches based on character voices"#,
                    char_display = char_display,
                    location_note = location_note,
                    tone_note = tone_note,
                ),
            ),
            PromptMessage::new_text(
                PromptMessageRole::User,
                format!(
                    "Plan a scene with {}{}{}.

Start by gathering the dynamics between these characters.",
                    char_display, location_note, tone_note
                ),
            ),
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                format!(
                    r#"I'll run the scene planning workflow:

**Step 1: Gather Pairwise Dynamics**
Use the `query` tool to get the scene plan:
```json
{{"operation": "scene_planning", "character_ids": [{char_ids}]}}
```

**Step 2: Deep-Dive Character States**
For each character, check their current knowledge and perceptions:
```json
{{"operation": "character_dossier", "character_id": "{first_char}"}}
```
Repeat for each character to understand their individual states.

**Step 3: Identify Tension Leverage Points**
From the scene plan results, identify:
- **Knowledge asymmetries**: What does each character know that others don't?
- **Perception gaps**: How do they misread each other?
- **Emotional stakes**: What does each character want from this encounter?

**Step 4: Check Universe Constraints**
```json
{{"operation": "list_facts", "category": "all"}}
```
Ensure the scene respects established facts and world rules.

**Step 5: Design Scene Beats**
Based on gathered data, I'll suggest:

| Beat | What Happens | Tension Driver | Character Focus |
|------|-------------|----------------|-----------------|
| Opening | Characters arrive/discover each other | Setup | Establish POV |
| Rising | Core interaction/revelation | Knowledge gap or perception gap | Active character |
| Pivot | Key moment of change/decision | Irony or conflict peak | All |
| Resolution | Consequences emerge | New knowledge state | Affected characters |

**Step 6: Dialogue Guidance**
For each character, I'll note:
- What they know vs. what they reveal
- How their perception of others colors their speech
- Emotional register based on relationship tensions
{location_note_detail}
{tone_note_detail}

Let me start by gathering the scene dynamics."#,
                    char_ids = char_ids_array,
                    first_char = char_list.first().unwrap_or(&"character:unknown"),
                    location_note_detail = match location_id {
                        Some(loc) => format!(
                            "\n**Location Context:**\nI'll also look up {} to factor in setting-specific details.",
                            loc
                        ),
                        None => String::new(),
                    },
                    tone_note_detail = match tone {
                        Some(t) => format!(
                            "\n**Tone Target:** {}\nI'll calibrate suggestions to match this tone.",
                            t
                        ),
                        None => String::new(),
                    },
                ),
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_planning_minimal() {
        let mut args = Map::new();
        args.insert(
            "character_ids".to_string(),
            Value::String("character:alice,character:bob".to_string()),
        );

        let result = get_scene_planning_prompt(Some(args));
        assert!(result.description.is_some());
        assert_eq!(result.messages.len(), 3);
        let desc = result.description.unwrap();
        assert!(desc.contains("alice"));
        assert!(desc.contains("bob"));
    }

    #[test]
    fn test_scene_planning_with_location_and_tone() {
        let mut args = Map::new();
        args.insert(
            "character_ids".to_string(),
            Value::String("character:alice,character:bob".to_string()),
        );
        args.insert(
            "location_id".to_string(),
            Value::String("location:tavern".to_string()),
        );
        args.insert("tone".to_string(), Value::String("tense".to_string()));

        let result = get_scene_planning_prompt(Some(args));
        let desc = result.description.unwrap();
        assert!(desc.contains("tavern"));
        assert!(desc.contains("tense"));
    }

    #[test]
    fn test_scene_planning_no_args() {
        let result = get_scene_planning_prompt(None);
        assert!(result.description.is_some());
        assert!(result
            .description
            .unwrap()
            .contains("no characters specified"));
    }
}
