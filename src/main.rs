//! Narra - Narrative intelligence engine for fiction writing
//!
//! Usage:
//!   narra mcp                    Start MCP server on stdio
//!   narra find "query"           Search across all entities
//!   narra get <name_or_id>       Get any entity by name or ID
//!   narra list characters        List all characters
//!   narra world status           World overview dashboard
//!   narra --help                 Show all commands

use anyhow::Result;
use clap::Parser;

use narra::cli::output::{DetailLevel, OutputMode};
use narra::cli::{Cli, Commands};
use narra::init::AppContext;
use narra::mcp::server::run_mcp_server;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Tracing to stderr (safe for MCP stdio transport)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("narra=info".parse()?),
        )
        .init();

    let mode = OutputMode::from_flags(cli.json, cli.md);
    let detail = DetailLevel::from_flags(cli.brief, cli.full);
    let no_semantic = cli.no_semantic;

    match &cli.command {
        Commands::Mcp => {
            let ctx = AppContext::new(cli.data_path.clone()).await?;
            run_mcp_server(ctx).await?;
        }
        cmd => {
            let ctx = AppContext::new(cli.data_path.clone()).await?;
            narra::cli::execute(cmd, &ctx, mode, detail, no_semantic).await?;
        }
    }

    Ok(())
}
