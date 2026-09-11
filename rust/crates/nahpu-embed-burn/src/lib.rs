use anyhow::Result;
use nahpu_embed_core::{Embedding, TextEmbedder, TextRole};

#[cfg(has_model)]
#[allow(clippy::all, clippy::pedantic, dead_code)]
mod model {
    include!(concat!(env!("OUT_DIR"), "/model/all-MiniLM-L6-v2.rs"));
}

#[cfg(all(
    feature = "flex",
    not(any(feature = "metal", feature = "vulkan", feature = "wgpu"))
))]
pub type SelectedBackend = burn::backend::Flex;
#[cfg(feature = "metal")]
pub type SelectedBackend = burn::backend::Metal;
#[cfg(all(feature = "vulkan", not(feature = "metal")))]
pub type SelectedBackend = burn::backend::Vulkan;
#[cfg(all(feature = "wgpu", not(any(feature = "metal", feature = "vulkan"))))]
pub type SelectedBackend = burn::backend::Wgpu;

pub const BACKEND_NAME: &str = if cfg!(feature = "metal") {
    "burn-metal"
} else if cfg!(feature = "vulkan") {
    "burn-vulkan"
} else if cfg!(feature = "wgpu") {
    "burn-wgpu"
} else {
    "burn-flex"
};

pub const HAS_MODEL: bool = cfg!(has_model);

pub const BASELINE_DIMENSIONS: usize = 384;

pub fn baseline_weights_path() -> Option<&'static str> {
    #[cfg(has_model)]
    return Some(env!("BASELINE_WEIGHTS"));
    #[cfg(not(has_model))]
    return None;
}

pub fn baseline_tokenizer_path() -> Option<&'static str> {
    #[cfg(has_model)]
    return Some(env!("BASELINE_TOKENIZER"));
    #[cfg(not(has_model))]
    return None;
}

#[cfg(has_model)]
pub use with_model::BurnEmbedder;

#[cfg(not(has_model))]
pub struct BurnEmbedder;

#[cfg(not(has_model))]
impl BurnEmbedder {
    pub fn from_bytes(_weights: &[u8], _tokenizer_json: &[u8]) -> Result<Self> {
        anyhow::bail!("no model compiled in — run scripts/fetch-model.sh, then rebuild")
    }
}

#[cfg(not(has_model))]
impl TextEmbedder for BurnEmbedder {
    fn name(&self) -> &str {
        BACKEND_NAME
    }

    fn dimensions(&self) -> usize {
        BASELINE_DIMENSIONS
    }

    fn embed(&self, _text: &str, _role: TextRole) -> Result<Embedding> {
        anyhow::bail!("no model compiled in — run scripts/fetch-model.sh, then rebuild")
    }
}

#[cfg(has_model)]
mod with_model {
    use super::*;

    use burn::prelude::*;
    use burn::tensor::Bytes;
    use tokenizers::Tokenizer;

    use crate::model::Model;

    pub struct BurnEmbedder {
        model: Model<SelectedBackend>,
        tokenizer: Tokenizer,
        device: Device<SelectedBackend>,
        name: String,
    }

    impl BurnEmbedder {
        pub fn from_bytes(weights: &[u8], tokenizer_json: &[u8]) -> Result<Self> {
            let device = Default::default();
            let model = Model::from_bytes(Bytes::from_elems(weights.to_vec()), &device);
            let tokenizer = Tokenizer::from_bytes(tokenizer_json)
                .map_err(|e| anyhow::anyhow!("could not read tokenizer: {e}"))?;

            Ok(Self {
                model,
                tokenizer,
                device,
                name: format!("{BACKEND_NAME}/all-MiniLM-L6-v2"),
            })
        }

        fn encode(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
            let encoded = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("could not tokenize: {e}"))?;

            let ids = encoded.get_ids().iter().map(|&id| id as i64).collect();
            let mask = encoded
                .get_attention_mask()
                .iter()
                .map(|&m| m as i64)
                .collect();
            Ok((ids, mask))
        }

        fn run_forward(&self, ids: &[i64], mask: &[i64]) -> Tensor<SelectedBackend, 3> {
            let shape = [1, ids.len()];
            let input_ids = self.int_tensor(ids, shape);
            let attention_mask = self.int_tensor(mask, shape);
            let token_type_ids = self.int_tensor(&vec![0; ids.len()], shape);

            self.model
                .forward(input_ids, attention_mask, token_type_ids)
        }

        fn int_tensor(&self, values: &[i64], shape: [usize; 2]) -> Tensor<SelectedBackend, 2, Int> {
            Tensor::from_data(TensorData::new(values.to_vec(), shape), &self.device)
        }
    }

    impl TextEmbedder for BurnEmbedder {
        fn name(&self) -> &str {
            &self.name
        }

        fn dimensions(&self) -> usize {
            BASELINE_DIMENSIONS
        }

        fn embed(&self, text: &str, _role: TextRole) -> Result<Embedding> {
            let (ids, mask) = self.encode(text)?;
            let hidden_states = self.run_forward(&ids, &mask);

            let pooled = mean_pool_over_unmasked_tokens(hidden_states, &mask, &self.device);
            Ok(normalise_to_unit_length(pooled))
        }
    }

    fn mean_pool_over_unmasked_tokens(
        hidden_states: Tensor<SelectedBackend, 3>,
        mask: &[i64],
        device: &Device<SelectedBackend>,
    ) -> Vec<f32> {
        let [_, tokens, width] = hidden_states.dims();

        let weights: Vec<f32> = mask.iter().map(|&m| m as f32).collect();
        let unmasked_count = weights.iter().sum::<f32>().max(1.0);

        let mask_column =
            Tensor::<SelectedBackend, 2>::from_data(TensorData::new(weights, [tokens, 1]), device);
        let weighted = hidden_states.squeeze_dim::<2>(0) * mask_column.repeat_dim(1, width);

        weighted
            .sum_dim(0)
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("pooled tensor is f32")
            .into_iter()
            .map(|total| total / unmasked_count)
            .collect()
    }

    fn normalise_to_unit_length(vector: Vec<f32>) -> Vec<f32> {
        let magnitude = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude == 0.0 {
            return vector;
        }
        vector.into_iter().map(|x| x / magnitude).collect()
    }
}
