//! Narra - World state management for fiction writing
//!
//! Usage:
//!   narra mcp                    Start MCP server on stdio
//!   narra character list         List characters
//!   narra search "query"         Global search
//!   narra --help                 Show all commands

use anyhow::Result;
use clap::Parser;

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

    match &cli.command {
        Commands::Mcp => {
            let ctx = AppContext::new(cli.data_path.clone()).await?;
            run_mcp_server(ctx).await?;
        }
        cmd => {
            let ctx = AppContext::new(cli.data_path.clone()).await?;
            narra::cli::execute(cmd, &ctx, cli.json).await?;
        }
    }

    Ok(())
}
