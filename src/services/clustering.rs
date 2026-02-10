//! Thematic clustering service for discovering emergent themes in stories.
//!
//! Groups entities by semantic similarity using their embeddings and K-means clustering.

use crate::db::connection::NarraDb;
use async_trait::async_trait;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::{Array1, Array2};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;

use crate::services::EntityType;
use crate::NarraError;

/// A member of a thematic cluster.
#[derive(Debug, Clone, Serialize)]
pub struct ClusterMember {
    pub entity_id: String,
    pub entity_type: String,
    pub name: String,
    pub centrality: f32,
}

/// A thematic cluster representing a story theme.
#[derive(Debug, Clone, Serialize)]
pub struct ThemeCluster {
    pub cluster_id: usize,
    pub label: String,
    pub members: Vec<ClusterMember>,
    pub member_count: usize,
}

/// Result of thematic clustering analysis.
#[derive(Debug, Clone, Serialize)]
pub struct ClusteringResult {
    pub clusters: Vec<ThemeCluster>,
    pub total_entities: usize,
    pub entities_without_embeddings: usize,
}

// ---------------------------------------------------------------------------
// Data provider trait
// ---------------------------------------------------------------------------

/// An entity with its embedding vector for clustering.
#[derive(Debug, Clone)]
pub struct EmbeddedEntity {
    pub id: String,
    pub entity_type: String,
    pub name: String,
    pub embedding: Vec<f32>,
}

/// Data access abstraction for the clustering service.
#[async_trait]
pub trait ClusteringDataProvider: Send + Sync {
    /// Get entities with embeddings for the given types.
    /// Returns (entities_with_embeddings, total_entity_count).
    async fn get_entities_with_embeddings(
        &self,
        entity_types: &[EntityType],
    ) -> Result<(Vec<EmbeddedEntity>, usize), NarraError>;
}

/// SurrealDB implementation of ClusteringDataProvider.
pub struct SurrealClusteringDataProvider {
    db: Arc<NarraDb>,
}

impl SurrealClusteringDataProvider {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl ClusteringDataProvider for SurrealClusteringDataProvider {
    async fn get_entities_with_embeddings(
        &self,
        entity_types: &[EntityType],
    ) -> Result<(Vec<EmbeddedEntity>, usize), NarraError> {
        let mut entity_data = Vec::new();
        let mut total_entities = 0;

        for entity_type in entity_types {
            if !entity_type.has_embeddings() {
                continue;
            }

            let table = entity_type.table_name();
            let name_field = match entity_type {
                EntityType::Event | EntityType::Scene => "title",
                _ => "name",
            };

            let query_str = format!(
                r#"SELECT id, '{table}' AS entity_type, {name_field} AS name, embedding
                   FROM {table}
                   WHERE embedding IS NOT NONE"#,
                table = table,
                name_field = name_field
            );

            let mut response = self.db.query(&query_str).await?;

            #[derive(serde::Deserialize)]
            struct EntityWithEmbedding {
                id: surrealdb::RecordId,
                entity_type: String,
                name: String,
                embedding: Vec<f32>,
            }

            let entities: Vec<EntityWithEmbedding> = response.take(0).unwrap_or_default();

            let count_query = format!("SELECT count() FROM {} GROUP ALL", table);
            let mut count_response = self.db.query(&count_query).await?;

            #[derive(serde::Deserialize)]
            struct CountResult {
                count: i64,
            }

            let counts: Vec<CountResult> = count_response.take(0).unwrap_or_default();
            if let Some(count_result) = counts.first() {
                total_entities += count_result.count as usize;
            }

            for entity in entities {
                entity_data.push(EmbeddedEntity {
                    id: entity.id.to_string(),
                    entity_type: entity.entity_type,
                    name: entity.name,
                    embedding: entity.embedding,
                });
            }
        }

        Ok((entity_data, total_entities))
    }
}

// ---------------------------------------------------------------------------
// ClusteringService
// ---------------------------------------------------------------------------

/// Service for discovering thematic clusters in story entities.
pub struct ClusteringService {
    data: Arc<dyn ClusteringDataProvider>,
}

impl ClusteringService {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self {
            data: Arc::new(SurrealClusteringDataProvider::new(db)),
        }
    }

    pub fn with_provider(data: Arc<dyn ClusteringDataProvider>) -> Self {
        Self { data }
    }

    /// Discover thematic clusters by analyzing entity embeddings.
    pub async fn discover_themes(
        &self,
        entity_types: Vec<EntityType>,
        num_themes: Option<usize>,
    ) -> Result<ClusteringResult, NarraError> {
        let (entity_data, total_entities) = self
            .data
            .get_entities_with_embeddings(&entity_types)
            .await?;

        let entities_with_embeddings = entity_data.len();
        let entities_without_embeddings = total_entities.saturating_sub(entities_with_embeddings);

        if entities_with_embeddings < 3 {
            return Err(NarraError::Database(format!(
                "Insufficient entities for clustering: {} entities with embeddings (need at least 3)",
                entities_with_embeddings
            )));
        }

        let num_clusters = if let Some(n) = num_themes {
            n.max(2).min(entities_with_embeddings - 1)
        } else {
            let auto = ((entities_with_embeddings as f32 / 2.0).sqrt()).ceil() as usize;
            auto.max(2).min(entities_with_embeddings - 1)
        };

        // Determine embedding dimensions from first entity
        let num_dims = entity_data[0].embedding.len();
        let mut matrix_data = Vec::with_capacity(entities_with_embeddings * num_dims);

        for entity in &entity_data {
            if entity.embedding.len() != num_dims {
                return Err(NarraError::Database(format!(
                    "Invalid embedding dimension: expected {}, got {}",
                    num_dims,
                    entity.embedding.len()
                )));
            }
            for &value in &entity.embedding {
                matrix_data.push(value as f64);
            }
        }

        let embedding_matrix =
            Array2::from_shape_vec((entities_with_embeddings, num_dims), matrix_data).map_err(
                |e| NarraError::Database(format!("Failed to create embedding matrix: {}", e)),
            )?;

        let dataset = DatasetBase::new(
            embedding_matrix.clone(),
            Array1::from_elem(entities_with_embeddings, ()),
        );

        let model = KMeans::params(num_clusters)
            .max_n_iterations(300)
            .tolerance(1e-4)
            .fit(&dataset)
            .map_err(|e| NarraError::Database(format!("K-means clustering failed: {}", e)))?;

        let predictions = model.predict(&dataset);
        let cluster_assignments: Vec<usize> = predictions.iter().cloned().collect();

        let centroids = model.centroids();

        let mut clusters: HashMap<usize, Vec<(String, String, String, f32)>> = HashMap::new();

        for (idx, cluster_id) in cluster_assignments.iter().enumerate() {
            let entity = &entity_data[idx];
            let centroid = centroids.row(*cluster_id);

            let mut distance_sq = 0.0_f64;
            for (i, &emb_val) in entity.embedding.iter().enumerate() {
                let diff = (emb_val as f64) - centroid[i];
                distance_sq += diff * diff;
            }
            let distance = distance_sq.sqrt();

            let centrality = 1.0 / (1.0 + distance);

            clusters.entry(*cluster_id).or_default().push((
                entity.id.clone(),
                entity.entity_type.clone(),
                entity.name.clone(),
                centrality as f32,
            ));
        }

        let mut theme_clusters: Vec<ThemeCluster> = clusters
            .into_iter()
            .map(|(cluster_id, mut members_data)| {
                members_data
                    .sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

                let members: Vec<ClusterMember> = members_data
                    .iter()
                    .map(|(id, entity_type, name, centrality)| ClusterMember {
                        entity_id: id.clone(),
                        entity_type: entity_type.clone(),
                        name: name.clone(),
                        centrality: *centrality,
                    })
                    .collect();

                let label_parts: Vec<String> =
                    members.iter().take(3).map(|m| m.name.clone()).collect();
                let label = label_parts.join(", ");

                let member_count = members.len();

                ThemeCluster {
                    cluster_id,
                    label,
                    members,
                    member_count,
                }
            })
            .collect();

        theme_clusters.sort_by(|a, b| b.member_count.cmp(&a.member_count));

        Ok(ClusteringResult {
            clusters: theme_clusters,
            total_entities,
            entities_without_embeddings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockClusteringDataProvider {
        entities: Vec<EmbeddedEntity>,
        total_count: usize,
    }

    #[async_trait]
    impl ClusteringDataProvider for MockClusteringDataProvider {
        async fn get_entities_with_embeddings(
            &self,
            _entity_types: &[EntityType],
        ) -> Result<(Vec<EmbeddedEntity>, usize), NarraError> {
            Ok((self.entities.clone(), self.total_count))
        }
    }

    fn make_entity(id: &str, name: &str, embedding: Vec<f32>) -> EmbeddedEntity {
        EmbeddedEntity {
            id: id.to_string(),
            entity_type: "character".to_string(),
            name: name.to_string(),
            embedding,
        }
    }

    #[tokio::test]
    async fn test_clustering_insufficient_entities() {
        let provider = MockClusteringDataProvider {
            entities: vec![
                make_entity("a", "A", vec![0.1; 4]),
                make_entity("b", "B", vec![0.2; 4]),
            ],
            total_count: 2,
        };
        let service = ClusteringService::with_provider(Arc::new(provider));
        let result = service
            .discover_themes(vec![EntityType::Character], None)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Insufficient"));
    }

    #[tokio::test]
    async fn test_clustering_distinct_clusters() {
        // Two obvious clusters: entities near [1,0,0,0] and entities near [0,0,0,1]
        let provider = MockClusteringDataProvider {
            entities: vec![
                make_entity("a1", "ClusterA1", vec![1.0, 0.0, 0.0, 0.0]),
                make_entity("a2", "ClusterA2", vec![0.9, 0.1, 0.0, 0.0]),
                make_entity("a3", "ClusterA3", vec![0.95, 0.05, 0.0, 0.0]),
                make_entity("b1", "ClusterB1", vec![0.0, 0.0, 0.0, 1.0]),
                make_entity("b2", "ClusterB2", vec![0.0, 0.0, 0.1, 0.9]),
                make_entity("b3", "ClusterB3", vec![0.0, 0.0, 0.05, 0.95]),
            ],
            total_count: 6,
        };
        let service = ClusteringService::with_provider(Arc::new(provider));
        let result = service
            .discover_themes(vec![EntityType::Character], Some(2))
            .await
            .unwrap();
        assert_eq!(result.clusters.len(), 2);
        assert_eq!(result.total_entities, 6);
        assert_eq!(result.entities_without_embeddings, 0);

        // Each cluster should have 3 members
        let mut sizes: Vec<usize> = result.clusters.iter().map(|c| c.member_count).collect();
        sizes.sort();
        assert_eq!(sizes, vec![3, 3]);
    }

    #[tokio::test]
    async fn test_clustering_single_cluster() {
        // All entities are similar — should still produce 2 clusters (minimum)
        let provider = MockClusteringDataProvider {
            entities: vec![
                make_entity("a", "A", vec![0.5, 0.5, 0.5, 0.5]),
                make_entity("b", "B", vec![0.5, 0.5, 0.5, 0.51]),
                make_entity("c", "C", vec![0.5, 0.5, 0.51, 0.5]),
            ],
            total_count: 3,
        };
        let service = ClusteringService::with_provider(Arc::new(provider));
        let result = service
            .discover_themes(vec![EntityType::Character], None)
            .await
            .unwrap();
        assert!(!result.clusters.is_empty());
        let total_members: usize = result.clusters.iter().map(|c| c.member_count).sum();
        assert_eq!(total_members, 3);
    }

    #[tokio::test]
    async fn test_clustering_entities_without_embeddings_count() {
        let provider = MockClusteringDataProvider {
            entities: vec![
                make_entity("a", "A", vec![0.1; 4]),
                make_entity("b", "B", vec![0.2; 4]),
                make_entity("c", "C", vec![0.3; 4]),
            ],
            total_count: 5, // 2 entities without embeddings
        };
        let service = ClusteringService::with_provider(Arc::new(provider));
        let result = service
            .discover_themes(vec![EntityType::Character], None)
            .await
            .unwrap();
        assert_eq!(result.entities_without_embeddings, 2);
    }
}
