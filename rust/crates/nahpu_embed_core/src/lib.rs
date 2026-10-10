use anyhow::Result;

pub mod preprocess;
pub mod zero_shot;

pub use preprocess::PreprocessConfig;

pub type Embedding = Vec<f32>;

pub const PARITY_THRESHOLD: f32 = 0.999;

/// Nomic models need a different prefix for queries than for stored text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextRole {
    Query,
    Document,
}

pub trait TextEmbedder {
    fn name(&self) -> &str;

    fn dimensions(&self) -> usize;

    fn embed(&self, text: &str, role: TextRole) -> Result<Embedding>;

    fn embed_batch(&self, texts: &[String], role: TextRole) -> Result<Vec<Embedding>> {
        texts.iter().map(|text| self.embed(text, role)).collect()
    }
}

pub trait ImageEmbedder {
    fn name(&self) -> &str;

    fn dimensions(&self) -> usize;

    fn preprocess_config(&self) -> PreprocessConfig;

    /// Takes the normalised CHW tensor that `preprocess` produces.
    fn embed_pixels(&self, pixels: &[f32]) -> Result<Embedding>;

    fn embed_image(&self, encoded_image: &[u8]) -> Result<Embedding> {
        let pixels = preprocess::preprocess(encoded_image, &self.preprocess_config())?;
        self.embed_pixels(&pixels)
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let magnitude_a = magnitude(a);
    let magnitude_b = magnitude(b);

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }
    dot_product / (magnitude_a * magnitude_b)
}

pub fn matches_known_good(candidate: &[f32], known_good: &[f32]) -> bool {
    cosine_similarity(candidate, known_good) > PARITY_THRESHOLD
}

pub fn normalise_to_unit_length(vector: Vec<f32>) -> Vec<f32> {
    let length = magnitude(&vector);
    if length == 0.0 {
        return vector;
    }
    vector.into_iter().map(|x| x / length).collect()
}

fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_are_cosine_one() {
        let v = vec![0.1, 0.2, 0.3, 0.4];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_has_no_similarity() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn parity_holds_for_near_identical_vectors() {
        assert!(matches_known_good(&[1.0, 2.0, 3.0], &[1.001, 2.0, 2.999]));
    }

    #[test]
    fn parity_fails_for_divergent_vectors() {
        assert!(!matches_known_good(&[1.0, 0.0], &[0.0, 1.0]));
    }
}
