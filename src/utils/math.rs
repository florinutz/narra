//! Shared mathematical utilities for vector operations.

/// Compute cosine similarity between two vectors.
/// Returns dot(a,b) / (norm(a) * norm(b)), or 0.0 if either vector has zero norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Element-wise vector subtraction: a - b.
pub fn vector_subtract(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Element-wise vector addition: a + b.
pub fn vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Scale a vector by a scalar: s * v.
pub fn vector_scale(v: &[f32], s: f32) -> Vec<f32> {
    v.iter().map(|x| x * s).collect()
}

/// Normalize a vector to unit length. Returns zero vector if input has zero norm.
pub fn vector_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        vec![0.0; v.len()]
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Compute the midpoint of two vectors: (a + b) / 2.
pub fn vector_midpoint(a: &[f32], b: &[f32]) -> Vec<f32> {
    vector_scale(&vector_add(a, b), 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical vectors should have similarity 1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "Orthogonal vectors should have similarity 0.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - (-1.0)).abs() < 1e-6,
            "Opposite vectors should have similarity -1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Zero vector should yield 0.0");
    }

    #[test]
    fn test_cosine_similarity_known_angle() {
        // 45-degree angle: cos(45) = 1/sqrt(2)
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5,
            "Expected ~0.7071, got {sim}",
        );
    }

    #[test]
    fn test_cosine_similarity_high_dimensional() {
        let a = vec![0.1; 384];
        let b = vec![0.1; 384];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical high-dim vectors: got {sim}"
        );
    }

    #[test]
    fn test_vector_subtract_basic() {
        let a = vec![3.0, 5.0, 7.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(vector_subtract(&a, &b), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vector_subtract_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert_eq!(vector_subtract(&a, &a), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_vector_subtract_negative_result() {
        let a = vec![1.0, 1.0];
        let b = vec![3.0, 5.0];
        assert_eq!(vector_subtract(&a, &b), vec![-2.0, -4.0]);
    }

    #[test]
    fn test_vector_subtract_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(vector_subtract(&a, &b), Vec::<f32>::new());
    }

    #[test]
    fn test_trajectory_similarity_same_direction() {
        let a_first = vec![1.0, 0.0, 0.0];
        let a_last = vec![1.0, 1.0, 0.0];
        let b_first = vec![0.0, 1.0, 0.0];
        let b_last = vec![0.0, 2.0, 0.0];
        let delta_a = vector_subtract(&a_last, &a_first);
        let delta_b = vector_subtract(&b_last, &b_first);
        let traj_sim = cosine_similarity(&delta_a, &delta_b);
        assert!(
            (traj_sim - 1.0).abs() < 1e-5,
            "Same-direction trajectories should have sim=1.0, got {traj_sim}",
        );
    }

    #[test]
    fn test_vector_add_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(vector_add(&a, &b), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_scale_double() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(vector_scale(&v, 2.0), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vector_scale_zero() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(vector_scale(&v, 0.0), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_vector_normalize_unit() {
        let v = vec![3.0, 4.0];
        let n = vector_normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_normalize_zero() {
        let v = vec![0.0, 0.0];
        assert_eq!(vector_normalize(&v), vec![0.0, 0.0]);
    }

    #[test]
    fn test_vector_midpoint() {
        let a = vec![0.0, 0.0];
        let b = vec![4.0, 6.0];
        assert_eq!(vector_midpoint(&a, &b), vec![2.0, 3.0]);
    }

    #[test]
    fn test_trajectory_similarity_opposite_direction() {
        let a_first = vec![0.0, 0.0];
        let a_last = vec![1.0, 0.0];
        let b_first = vec![0.0, 0.0];
        let b_last = vec![-1.0, 0.0];
        let delta_a = vector_subtract(&a_last, &a_first);
        let delta_b = vector_subtract(&b_last, &b_first);
        let traj_sim = cosine_similarity(&delta_a, &delta_b);
        assert!(
            (traj_sim - (-1.0)).abs() < 1e-5,
            "Opposite trajectories should have sim=-1.0, got {traj_sim}",
        );
    }
}
