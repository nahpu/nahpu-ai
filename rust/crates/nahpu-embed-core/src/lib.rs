//! Shared abstractions for NAHPU embedding experiments.
//!
//! Nothing here depends on an inference framework. Engines live behind the
//! traits below, so an experiment, a benchmark, or the eventual `nahpu_dl`
//! crate stays the same whichever framework produced the vectors. Swapping
//! Burn for something else is one new impl, not a rewrite.

use anyhow::Result;

/// A single embedding vector.
pub type Embedding = Vec<f32>;

/// What a piece of text is being embedded *for*.
///
/// Nomic models require a task prefix on the input, and a search query must be
/// prefixed differently from a stored document or retrieval quality drops. The
/// engine owns the prefix strings; callers only say which role applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextRole {
    /// Text the user is searching with.
    Query,
    /// Text being indexed for later retrieval.
    Document,
}

/// Text to vector.
pub trait TextEmbedder {
    /// Short identifier used in reports, e.g. `"burn-flex/nomic-text-v1.5"`.
    fn name(&self) -> &str;

    /// Dimensionality of the vectors this engine returns.
    fn dim(&self) -> usize;

    fn embed(&self, text: &str, role: TextRole) -> Result<Embedding>;

    /// Embed a batch. The default loops; override where the framework batches
    /// more efficiently.
    fn embed_batch(&self, texts: &[String], role: TextRole) -> Result<Vec<Embedding>> {
        texts.iter().map(|t| self.embed(t, role)).collect()
    }
}

/// Image to vector.
///
/// Kept separate from [`TextEmbedder`] because an engine may support one and
/// not the other. A model whose image and text vectors share a space (the
/// nomic pair) implements both and reports the same [`TextEmbedder::dim`].
pub trait ImageEmbedder {
    fn name(&self) -> &str;

    fn dim(&self) -> usize;

    /// `bytes` is an encoded image (PNG, JPEG); the engine decodes it.
    fn embed_image(&self, bytes: &[u8]) -> Result<Embedding>;
}

/// Cosine similarity between two vectors.
///
/// Used to gate every engine: a new backend has to reproduce a known-good
/// vector before any speed number from it is worth reading.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_identical_is_one() {
        let v = vec![0.1, 0.2, 0.3, 0.4];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_handles_zero_vector() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }
}
