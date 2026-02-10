use std::path::Path;

use serde::{Deserialize, Serialize};
use surrealdb::engine::any::Any;
use surrealdb::opt::capabilities::Capabilities;
use surrealdb::Surreal;

use crate::NarraError;

/// Unified database handle type. Works with both embedded and remote SurrealDB.
pub type NarraDb = Surreal<Any>;

fn default_namespace() -> String {
    "narra".to_string()
}

fn default_database() -> String {
    "world".to_string()
}

/// Database connection configuration.
/// Loaded from `{data_path}/database.toml`, env vars, or defaults to embedded.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum DbConfig {
    /// Embedded RocksDB (default, current behavior). Single-process access.
    Embedded {
        /// Overrides the default RocksDB path (`{data_path}` is implicit)
        #[serde(default)]
        path: Option<String>,
    },
    /// Remote SurrealDB server via WebSocket. Supports concurrent access.
    Remote {
        /// WebSocket endpoint (e.g. `ws://127.0.0.1:8000`, `wss://host:port`)
        endpoint: String,
        /// Username (can also be set via `NARRA_DB_USER` env var)
        #[serde(default)]
        username: Option<String>,
        /// Password (can also be set via `NARRA_DB_PASS` env var)
        #[serde(default)]
        password: Option<String>,
        /// SurrealDB namespace (default: `"narra"`)
        #[serde(default = "default_namespace")]
        namespace: String,
        /// SurrealDB database (default: `"world"`)
        #[serde(default = "default_database")]
        database: String,
    },
}

impl Default for DbConfig {
    fn default() -> Self {
        Self::Embedded { path: None }
    }
}

/// Load database config with priority:
/// 1. `{data_path}/database.toml` file
/// 2. `NARRA_DB_URL` env var → creates `Remote` config
/// 3. Default → `Embedded { path: None }`
pub fn load_db_config(data_path: &Path) -> DbConfig {
    // Try file first
    let config_path = data_path.join("database.toml");
    if config_path.exists() {
        match std::fs::read_to_string(&config_path) {
            Ok(contents) => match toml::from_str::<DbConfig>(&contents) {
                Ok(config) => {
                    tracing::info!("Loaded database config from {}", config_path.display());
                    return config;
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse {}: {}. Using default.",
                        config_path.display(),
                        e
                    );
                }
            },
            Err(e) => {
                tracing::warn!(
                    "Failed to read {}: {}. Using default.",
                    config_path.display(),
                    e
                );
            }
        }
    }

    // Try NARRA_DB_URL env var
    if let Ok(url) = std::env::var("NARRA_DB_URL") {
        tracing::info!("Loaded database config from NARRA_DB_URL env");
        return DbConfig::Remote {
            endpoint: url,
            username: std::env::var("NARRA_DB_USER").ok(),
            password: std::env::var("NARRA_DB_PASS").ok(),
            namespace: default_namespace(),
            database: default_database(),
        };
    }

    DbConfig::default()
}

/// Initialize and connect to a SurrealDB database.
///
/// Supports both embedded RocksDB (single-process) and remote WebSocket
/// (concurrent access) modes, driven by `DbConfig`.
///
/// # Arguments
///
/// * `config` - Database connection configuration
/// * `data_path` - Base data directory (used as default RocksDB path for embedded mode)
pub async fn init_db(config: &DbConfig, data_path: &Path) -> Result<NarraDb, NarraError> {
    match config {
        DbConfig::Embedded { path } => {
            let db_path = path
                .as_deref()
                .map(String::from)
                .unwrap_or_else(|| data_path.to_string_lossy().into_owned());
            let surreal_config = surrealdb::opt::Config::new()
                .capabilities(Capabilities::all().with_all_experimental_features_allowed());
            let db =
                surrealdb::engine::any::connect((format!("rocksdb:{db_path}"), surreal_config))
                    .await?;
            db.use_ns("narra").use_db("world").await?;
            Ok(db)
        }
        DbConfig::Remote {
            endpoint,
            username,
            password,
            namespace,
            database,
        } => {
            let db = surrealdb::engine::any::connect(endpoint).await?;
            // Resolve credentials: config field > env var > default
            let user = username
                .clone()
                .or_else(|| std::env::var("NARRA_DB_USER").ok())
                .unwrap_or_else(|| "root".to_string());
            let pass = password
                .clone()
                .or_else(|| std::env::var("NARRA_DB_PASS").ok())
                .unwrap_or_else(|| "root".to_string());
            db.signin(surrealdb::opt::auth::Root {
                username: &user,
                password: &pass,
            })
            .await?;
            db.use_ns(namespace).use_db(database).await?;
            Ok(db)
        }
    }
}
