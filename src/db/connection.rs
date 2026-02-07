use surrealdb::engine::local::{Db, RocksDb};
use surrealdb::opt::capabilities::Capabilities;
use surrealdb::Surreal;

use crate::NarraError;

/// Initialize and connect to a SurrealDB database with RocksDB persistence.
///
/// This creates or opens a database at the specified path, using the "narra"
/// namespace and "world" database.
///
/// # Arguments
///
/// * `path` - Path to the RocksDB database directory
///
/// # Returns
///
/// A connected database handle ready for operations.
///
/// # Example
///
/// ```no_run
/// # use narra::db::connection::init_db;
/// # async fn example() -> Result<(), narra::NarraError> {
/// let db = init_db("./data/narra.db").await?;
/// # Ok(())
/// # }
/// ```
pub async fn init_db(path: &str) -> Result<Surreal<Db>, NarraError> {
    let config = surrealdb::opt::Config::new()
        .capabilities(Capabilities::all().with_all_experimental_features_allowed());
    let db = Surreal::new::<RocksDb>((path, config)).await?;
    db.use_ns("narra").use_db("world").await?;
    Ok(db)
}
