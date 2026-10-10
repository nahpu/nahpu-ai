use anyhow::Result;
use nahpu_embed_core::{Embedding, TextEmbedder};

pub fn load_minilm() -> Result<Box<dyn TextEmbedder>> {
    #[cfg(has_minilm)]
    return Ok(Box::new(with_model::MiniLm::load()?));
    #[cfg(not(has_minilm))]
    return Err(crate::missing_model("all-MiniLM-L6-v2"));
}

#[allow(dead_code)]
const DIMENSIONS: usize = 384;

#[cfg(has_minilm)]
#[allow(clippy::all, clippy::pedantic, dead_code)]
mod model {
    include!(concat!(env!("OUT_DIR"), "/model/all-MiniLM-L6-v2.rs"));
}

#[cfg(has_minilm)]
mod with_model {
    use super::*;

    use burn::prelude::*;
    use nahpu_embed_core::{normalise_to_unit_length, TextRole};
    use tokenizers::Tokenizer;

    use super::model::Model;
    use crate::{SelectedBackend, BACKEND_NAME};

    pub struct MiniLm {
        model: Model<SelectedBackend>,
        tokenizer: Tokenizer,
        device: Device<SelectedBackend>,
        name: String,
    }

    impl MiniLm {
        pub fn load() -> Result<Self> {
            let device = Default::default();
            let weights = crate::read_weights(env!("MINILM_WEIGHTS"))?;

            Ok(Self {
                model: Model::from_bytes(weights, &device),
                tokenizer: crate::read_tokenizer("all-MiniLM-L6-v2")?,
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

    impl TextEmbedder for MiniLm {
        fn name(&self) -> &str {
            &self.name
        }

        fn dimensions(&self) -> usize {
            DIMENSIONS
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
}
