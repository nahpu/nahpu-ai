use std::path::{Path, PathBuf};

use anyhow::Result;
use burn::prelude::*;

mod bioclip;
mod clip;
mod jina;
mod minilm;
mod nomic;
mod pe;

pub use bioclip::{load_bioclip_text, load_bioclip_vision, BIOCLIP_LOGIT_SCALE};
pub use clip::{load_clip, ClipModel, CLIP_MODEL_NAMES};
pub use jina::{load_jina_text, load_jina_vision, JINA_LOGIT_SCALE};
pub use minilm::load_minilm;
pub use nahpu_burn_support::BACKEND_NAME;
pub use nomic::{load_nomic_text, load_nomic_vision, NOMIC_LOGIT_SCALE};
pub use pe::{PeCoreSize, PE_CORE_SIZES};

// For the model modules; unused when no model is compiled in.
#[allow(unused_imports)]
use nahpu_burn_support::{read_weights, selected_device};

pub const HAS_MINILM: bool = cfg!(has_minilm);
pub const HAS_JINA: bool = cfg!(all(has_jina_text, has_jina_vision));
pub const HAS_BIOCLIP: bool = cfg!(all(has_bioclip_text, has_bioclip_vision));
pub const HAS_NOMIC: bool = cfg!(all(has_nomic_text, has_nomic_vision));

pub fn models_dir() -> &'static Path {
    Path::new(env!("NAHPU_MODELS_DIR"))
}

fn tokenizer_path(stem: &str) -> PathBuf {
    models_dir().join(format!("{stem}.tokenizer.json"))
}

fn read_tokenizer(stem: &str) -> Result<tokenizers::Tokenizer> {
    let path = tokenizer_path(stem);
    tokenizers::Tokenizer::from_file(&path)
        .map_err(|e| anyhow::anyhow!("could not read tokenizer {}: {e}", path.display()))
}

fn rows_of(output: Tensor<2>) -> Vec<Vec<f32>> {
    let [_, width] = output.dims();
    output
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .expect("output tensor is f32")
        .chunks(width)
        .map(<[f32]>::to_vec)
        .collect()
}

#[allow(dead_code)]
fn missing_model(stem: &str) -> anyhow::Error {
    anyhow::anyhow!("{stem} is not compiled in — run scripts/fetch-model.sh, then rebuild")
}
