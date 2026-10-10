use anyhow::Result;
use nahpu_embed_core::{ImageEmbedder, TextEmbedder};

/// exp of the trained `logit_scale` (4.5904) in jina-clip-v1's checkpoint.
pub const JINA_LOGIT_SCALE: f32 = 98.54;

#[allow(dead_code)]
const DIMENSIONS: usize = 768;
#[allow(dead_code)]
const MAX_TOKENS: usize = 512;

pub fn load_jina_text() -> Result<Box<dyn TextEmbedder>> {
    #[cfg(has_jina_text)]
    return Ok(Box::new(text::JinaText::load()?));
    #[cfg(not(has_jina_text))]
    return Err(crate::missing_model("jina-clip-v1-text"));
}

pub fn load_jina_vision() -> Result<Box<dyn ImageEmbedder>> {
    #[cfg(has_jina_vision)]
    return Ok(Box::new(vision::JinaVision::load()?));
    #[cfg(not(has_jina_vision))]
    return Err(crate::missing_model("jina-clip-v1-vision"));
}

#[cfg(has_jina_text)]
mod text {
    use super::*;

    use burn::prelude::*;
    use nahpu_embed_core::{normalise_to_unit_length, Embedding, TextRole};
    use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

    use crate::{SelectedBackend, BACKEND_NAME};

    #[allow(clippy::all, clippy::pedantic, dead_code)]
    mod model {
        include!(concat!(env!("OUT_DIR"), "/model/jina-clip-v1-text.rs"));
    }

    pub struct JinaText {
        model: model::Model<SelectedBackend>,
        tokenizer: Tokenizer,
        device: Device<SelectedBackend>,
        name: String,
    }

    impl JinaText {
        pub fn load() -> Result<Self> {
            let device = Default::default();
            let weights = crate::read_weights(env!("JINA_TEXT_WEIGHTS"))?;

            Ok(Self {
                model: model::Model::from_bytes(weights, &device),
                tokenizer: padded_and_truncated(crate::read_tokenizer("jina-clip-v1")?)?,
                device,
                name: format!("{BACKEND_NAME}/jina-clip-v1-text"),
            })
        }

        /// The graph builds its own attention mask from `input_ids != 0`, so
        /// padding with 0 leaves every row's vector unchanged.
        fn token_ids(&self, texts: &[String]) -> Result<(Vec<i64>, [usize; 2])> {
            let encodings = self
                .tokenizer
                .encode_batch(texts.to_vec(), true)
                .map_err(|e| anyhow::anyhow!("could not tokenize: {e}"))?;

            let longest = encodings.first().map_or(0, |e| e.get_ids().len());
            let ids = encodings
                .iter()
                .flat_map(|e| e.get_ids().iter().map(|&id| i64::from(id)))
                .collect();
            Ok((ids, [texts.len(), longest]))
        }
    }

    fn padded_and_truncated(mut tokenizer: Tokenizer) -> Result<Tokenizer> {
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: 0,
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("could not set truncation: {e}"))?;
        Ok(tokenizer)
    }

    impl TextEmbedder for JinaText {
        fn name(&self) -> &str {
            &self.name
        }

        fn dimensions(&self) -> usize {
            DIMENSIONS
        }

        fn embed(&self, text: &str, role: TextRole) -> Result<Embedding> {
            let mut rows = self.embed_batch(&[text.to_string()], role)?;
            Ok(rows.remove(0))
        }

        fn embed_batch(&self, texts: &[String], _role: TextRole) -> Result<Vec<Embedding>> {
            if texts.is_empty() {
                return Ok(Vec::new());
            }
            let (ids, shape) = self.token_ids(texts)?;
            let input_ids = Tensor::<SelectedBackend, 2, Int>::from_data(TensorData::new(ids, shape), &self.device);

            let output = self.model.forward(input_ids);
            Ok(crate::rows_of(output).into_iter().map(normalise_to_unit_length).collect())
        }
    }
}

#[cfg(has_jina_vision)]
mod vision {
    use super::*;

    use burn::prelude::*;
    use nahpu_embed_core::{normalise_to_unit_length, Embedding, PreprocessConfig};

    use crate::{SelectedBackend, BACKEND_NAME};

    #[allow(clippy::all, clippy::pedantic, dead_code)]
    mod model {
        include!(concat!(env!("OUT_DIR"), "/model/jina-clip-v1-vision.rs"));
    }

    pub struct JinaVision {
        model: model::Model<SelectedBackend>,
        device: Device<SelectedBackend>,
        name: String,
    }

    impl JinaVision {
        pub fn load() -> Result<Self> {
            let device = Default::default();
            let weights = crate::read_weights(env!("JINA_VISION_WEIGHTS"))?;

            Ok(Self {
                model: model::Model::from_bytes(weights, &device),
                device,
                name: format!("{BACKEND_NAME}/jina-clip-v1-vision"),
            })
        }
    }

    impl ImageEmbedder for JinaVision {
        fn name(&self) -> &str {
            &self.name
        }

        fn dimensions(&self) -> usize {
            DIMENSIONS
        }

        fn preprocess_config(&self) -> PreprocessConfig {
            PreprocessConfig::OPENAI_CLIP_224
        }

        fn embed_pixels(&self, pixels: &[f32]) -> Result<Embedding> {
            let size = self.preprocess_config().size as usize;
            anyhow::ensure!(
                pixels.len() == 3 * size * size,
                "expected a 3x{size}x{size} tensor, got {} values",
                pixels.len()
            );
            let input = Tensor::<SelectedBackend, 4>::from_data(
                TensorData::new(pixels.to_vec(), [1, 3, size, size]),
                &self.device,
            );

            let output = self.model.forward(input);
            Ok(normalise_to_unit_length(crate::rows_of(output).remove(0)))
        }
    }
}
