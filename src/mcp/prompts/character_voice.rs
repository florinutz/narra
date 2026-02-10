//! Character voice prompt for consistent dialogue generation.
//!
//! Provides psychological profile, knowledge state, and perception data
//! to guide the LLM in generating consistent character dialogue and monologue.

use rmcp::model::{GetPromptResult, PromptMessage, PromptMessageRole};
use serde_json::{Map, Value};

/// Get the character voice prompt.
///
/// # Arguments
/// - `character_id` (required): Character ID (e.g., "character:alice")
/// - `context` (optional): Scene or event context for the dialogue
pub fn get_character_voice_prompt(args: Option<Map<String, Value>>) -> GetPromptResult {
    let character_id = args
        .as_ref()
        .and_then(|a| a.get("character_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("(no character specified)");

    let context = args
        .as_ref()
        .and_then(|a| a.get("context"))
        .and_then(|v| v.as_str());

    let context_note = match context {
        Some(c) => format!(" in the context of: {}", c),
        None => String::new(),
    };

    GetPromptResult {
        description: Some(format!(
            "Generate voice profile for {}{}",
            character_id, context_note
        )),
        messages: vec![
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                format!(
                    r#"I'll build a voice profile for {character_id}{context_note} to help you write consistent dialogue.

**Voice Profile Components:**
1. Psychological foundation (personality, profile traits)
2. Knowledge state (what they know, believe, suspect)
3. Perception of others (how they see the people around them)
4. Speech patterns derived from all of the above"#,
                    character_id = character_id,
                    context_note = context_note,
                ),
            ),
            PromptMessage::new_text(
                PromptMessageRole::User,
                format!(
                    "Build a voice profile for {}{} so I can write dialogue that sounds like them.",
                    character_id, context_note
                ),
            ),
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                format!(
                    r#"I'll gather the data needed for {character_id}'s voice profile:

**Step 1: Character Foundation**
```json
{{"operation": "lookup", "entity_id": "{character_id}"}}
```
This gives us their name, roles, description, and profile traits (wounds, desires, contradictions, etc.).

**Step 2: Knowledge State**
```json
{{"operation": "character_dossier", "character_id": "{character_id}"}}
```
This reveals what they know, their blind spots, and false beliefs — all of which shape how they speak.

**Step 3: Character Voice Analysis**
```json
{{"operation": "character_voice", "character_id": "{character_id}"}}
```
This gives us their speech tendencies derived from their psychological profile.

**Step 4: Perception Context**
```json
{{"operation": "perception_matrix", "target_id": "{character_id}"}}
```
How others see them informs how they might respond to or anticipate others' reactions.

**Step 5: Synthesize Voice Guidelines**

From the gathered data, I'll produce:

### Vocabulary & Diction
- **Register**: Formal/informal/mixed based on role and personality
- **Knowledge-informed**: Words they'd use vs. avoid based on what they know
- **Emotional coloring**: How their profile traits flavor word choice

### Speech Patterns
- **Certainty level**: How confidently they assert things (informed by knowledge state)
- **Information management**: What they volunteer vs. hold back
- **Deflection patterns**: Topics they avoid based on wounds/secrets

### Dialogue Don'ts
- Things this character would NEVER say (based on knowledge gaps)
- Tones that contradict their established personality
- References they couldn't make (knowledge they don't have)

### Sample Lines
Based on the profile, I'll provide 3-5 example lines showing the voice in action.
{context_detail}

Let me start gathering the character data."#,
                    character_id = character_id,
                    context_detail = match context {
                        Some(c) => format!(
                            "\n**Scene Context:** {}\nI'll tailor the voice guidance to this specific situation.",
                            c
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
    fn test_character_voice_minimal() {
        let mut args = Map::new();
        args.insert(
            "character_id".to_string(),
            Value::String("character:alice".to_string()),
        );

        let result = get_character_voice_prompt(Some(args));
        assert!(result.description.is_some());
        assert_eq!(result.messages.len(), 3);
        assert!(result.description.unwrap().contains("alice"));
    }

    #[test]
    fn test_character_voice_with_context() {
        let mut args = Map::new();
        args.insert(
            "character_id".to_string(),
            Value::String("character:alice".to_string()),
        );
        args.insert(
            "context".to_string(),
            Value::String("confrontation with Bob at the tavern".to_string()),
        );

        let result = get_character_voice_prompt(Some(args));
        let desc = result.description.unwrap();
        assert!(desc.contains("confrontation"));
    }

    #[test]
    fn test_character_voice_no_args() {
        let result = get_character_voice_prompt(None);
        assert!(result
            .description
            .unwrap()
            .contains("no character specified"));
    }
}
