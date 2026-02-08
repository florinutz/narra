//! Output formatting infrastructure for CLI commands.

use colored::Colorize;
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Table};
use serde::Serialize;

/// Output mode for CLI commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    Human,
    Json,
    Markdown,
}

impl OutputMode {
    pub fn from_json_flag(json: bool) -> Self {
        if json {
            OutputMode::Json
        } else {
            OutputMode::Human
        }
    }

    pub fn from_flags(json: bool, md: bool) -> Self {
        if json {
            OutputMode::Json
        } else if md {
            OutputMode::Markdown
        } else {
            OutputMode::Human
        }
    }
}

/// Detail level for CLI output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetailLevel {
    Brief,
    Standard,
    Full,
}

impl DetailLevel {
    pub fn from_flags(brief: bool, full: bool) -> Self {
        if brief {
            DetailLevel::Brief
        } else if full {
            DetailLevel::Full
        } else {
            DetailLevel::Standard
        }
    }
}

/// Print a single item as pretty-printed JSON.
pub fn output_json<T: Serialize>(item: &T) {
    match serde_json::to_string_pretty(item) {
        Ok(json) => println!("{}", json),
        Err(e) => print_error(&format!("Failed to serialize to JSON: {}", e)),
    }
}

/// Print a list of items as a JSON array.
pub fn output_json_list<T: Serialize>(items: &[T]) {
    match serde_json::to_string_pretty(items) {
        Ok(json) => println!("{}", json),
        Err(e) => print_error(&format!("Failed to serialize to JSON: {}", e)),
    }
}

/// Print a formatted table with headers and rows.
pub fn print_table(headers: &[&str], rows: Vec<Vec<String>>) {
    if rows.is_empty() {
        println!("{}", "No results found.".dimmed());
        return;
    }

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(headers);

    for row in rows {
        table.add_row(row);
    }

    println!("{table}");
}

/// Print a success message.
pub fn print_success(msg: &str) {
    println!("{} {}", "OK".green().bold(), msg);
}

/// Print an error message to stderr.
pub fn print_error(msg: &str) {
    eprintln!("{} {}", "Error:".red().bold(), msg);
}

/// Print a bold section header.
pub fn print_header(title: &str) {
    println!("\n{}\n", title.bold());
}

/// Print a key-value pair line.
pub fn print_kv(key: &str, value: &str) {
    println!("  {}: {}", key.dimmed(), value);
}

/// Print a titled section with content.
pub fn print_section(title: &str, content: &str) {
    println!("\n{}", title.bold().underline());
    println!("{}", content);
}

/// Print a dimmed hint/suggestion message.
pub fn print_hint(msg: &str) {
    println!("{}", msg.dimmed());
}
