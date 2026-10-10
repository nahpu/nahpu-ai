use anyhow::Result;
use nahpu_embed_core::{ImageEmbedder, TextEmbedder};

/// exp of the trained `logit_scale` (4.6052) in BioCLIP's checkpoint.
pub const BIOCLIP_LOGIT_SCALE: f32 = 100.0;

#[allow(dead_code)]
const DIMENSIONS: usize = 512;
/// open_clip always feeds 77 tokens, padded with 0.
#[allow(dead_code)]
const CONTEXT_LENGTH: usize = 77;
#[allow(dead_code)]
const END_OF_TEXT: u32 = 49407;

pub fn load_bioclip_text() -> Result<Box<dyn TextEmbedder>> {
    #[cfg(has_bioclip_text)]
    return Ok(Box::new(text::BioClipText::load()?));
    #[cfg(not(has_bioclip_text))]
    return Err(crate::missing_model("bioclip-text"));
}

pub fn load_bioclip_vision() -> Result<Box<dyn ImageEmbedder>> {
    #[cfg(has_bioclip_vision)]
    return Ok(Box::new(vision::BioClipVision::load()?));
    #[cfg(not(has_bioclip_vision))]
    return Err(crate::missing_model("bioclip-vision"));
}

#[cfg(has_bioclip_text)]
mod text {
    use super::*;

    use burn::prelude::*;
    use nahpu_embed_core::{normalise_to_unit_length, Embedding, TextRole};
    use tokenizers::Tokenizer;

    use crate::{SelectedBackend, BACKEND_NAME};

    #[allow(clippy::all, clippy::pedantic, dead_code)]
    mod model {
        include!(concat!(env!("OUT_DIR"), "/model/bioclip-text.rs"));
    }

    pub struct BioClipText {
        model: model::Model<SelectedBackend>,
        tokenizer: Tokenizer,
        device: Device<SelectedBackend>,
        name: String,
    }

    impl BioClipText {
        pub fn load() -> Result<Self> {
            let device = Default::default();
            let weights = crate::read_weights(env!("BIOCLIP_TEXT_WEIGHTS"))?;

            Ok(Self {
                model: model::Model::from_bytes(weights, &device),
                tokenizer: crate::read_tokenizer("bioclip")?,
                device,
                name: format!("{BACKEND_NAME}/bioclip-text"),
            })
        }

        fn token_ids(&self, text: &str) -> Result<Vec<i64>> {
            let encoded = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("could not tokenize: {e}"))?;
            Ok(fit_to_context(encoded.get_ids()))
        }

        fn embed_one(&self, text: &str) -> Result<Embedding> {
            let input_ids = Tensor::<SelectedBackend, 2, Int>::from_data(
                TensorData::new(self.token_ids(text)?, [1, CONTEXT_LENGTH]),
                &self.device,
            );
            let output = self.model.forward(input_ids);
            Ok(normalise_to_unit_length(crate::rows_of(output).remove(0)))
        }
    }

    /// Mirrors open_clip.tokenize: cut to 77 keeping end-of-text last, pad with 0.
    fn fit_to_context(ids: &[u32]) -> Vec<i64> {
        let mut fitted: Vec<i64> = ids.iter().take(CONTEXT_LENGTH).map(|&id| i64::from(id)).collect();
        if ids.len() > CONTEXT_LENGTH {
            fitted[CONTEXT_LENGTH - 1] = i64::from(END_OF_TEXT);
        }
        fitted.resize(CONTEXT_LENGTH, 0);
        fitted
    }

    impl TextEmbedder for BioClipText {
        fn name(&self) -> &str {
            &self.name
        }

        fn dimensions(&self) -> usize {
            DIMENSIONS
        }

        fn embed(&self, text: &str, _role: TextRole) -> Result<Embedding> {
            self.embed_one(text)
        }

        /// The export is fixed at batch 1, so prompts go through one at a time.
        fn embed_batch(&self, texts: &[String], _role: TextRole) -> Result<Vec<Embedding>> {
            texts.iter().map(|text| self.embed_one(text)).collect()
        }
    }
}

#[cfg(has_bioclip_vision)]
mod vision {
    use super::*;

    use burn::prelude::*;
    use nahpu_embed_core::{normalise_to_unit_length, Embedding, PreprocessConfig};

    use crate::{SelectedBackend, BACKEND_NAME};

    #[allow(clippy::all, clippy::pedantic, dead_code)]
    mod model {
        include!(concat!(env!("OUT_DIR"), "/model/bioclip-vision.rs"));
    }

    pub struct BioClipVision {
        model: model::Model<SelectedBackend>,
        device: Device<SelectedBackend>,
        name: String,
    }

    impl BioClipVision {
        pub fn load() -> Result<Self> {
            let device = Default::default();
            let weights = crate::read_weights(env!("BIOCLIP_VISION_WEIGHTS"))?;

            Ok(Self {
                model: model::Model::from_bytes(weights, &device),
                device,
                name: format!("{BACKEND_NAME}/bioclip-vision"),
            })
        }
    }

    impl ImageEmbedder for BioClipVision {
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
