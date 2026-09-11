use anyhow::{bail, Result};
use nahpu_embed_core::{Embedding, TextRole};

#[cfg(has_model)]
mod model {
    include!(concat!(env!("OUT_DIR"), "/model/all_minilm_l6_v2.rs"));
}

#[cfg(all(
    feature = "flex",
    not(any(feature = "metal", feature = "vulkan", feature = "wgpu"))
))]
pub type Backend = burn::backend::Flex;
#[cfg(feature = "metal")]
pub type Backend = burn::backend::Metal;
#[cfg(all(feature = "vulkan", not(feature = "metal")))]
pub type Backend = burn::backend::Vulkan;
#[cfg(all(feature = "wgpu", not(any(feature = "metal", feature = "vulkan"))))]
pub type Backend = burn::backend::Wgpu;

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

pub struct BurnEmbedder {
    name: String,
}

impl BurnEmbedder {
    pub fn from_bytes(_weights: &[u8]) -> Result<Self> {
        if !HAS_MODEL {
            bail!("no model compiled in — run scripts/fetch-model.sh, then rebuild");
        }
        Ok(Self {
            name: format!("{BACKEND_NAME}/all-MiniLM-L6-v2"),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn embed(&self, _text: &str, _role: TextRole) -> Result<Embedding> {
        bail!("not implemented yet — phase 1 wires the forward pass")
    }
}
