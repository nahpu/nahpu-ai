//! Burn-backed embedding engines.
//!
//! The backend is chosen at compile time through cargo features and exposed as
//! [`Backend`], so the rest of the crate is written once against one type.

use anyhow::{bail, Result};
use nahpu_embed_core::{Embedding, TextRole};

#[cfg(has_model)]
mod model {
    include!(concat!(env!("OUT_DIR"), "/model/all_minilm_l6_v2.rs"));
}

/// The compiled-in backend.
///
/// `flex` is pure Rust, so it builds on every target with no system library.
/// GPU backends are opt-in; enabling one replaces the CPU type here.
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

/// Name of the active backend, for reports and for the CLI to print.
pub const BACKEND_NAME: &str = if cfg!(feature = "metal") {
    "burn-metal"
} else if cfg!(feature = "vulkan") {
    "burn-vulkan"
} else if cfg!(feature = "wgpu") {
    "burn-wgpu"
} else {
    "burn-flex"
};

/// Whether this build has a model compiled in.
///
/// False on a fresh clone until `scripts/fetch-model.sh` has run.
pub const HAS_MODEL: bool = cfg!(has_model);

/// A Burn embedding engine.
///
/// Construct it from weight bytes rather than a path. The caller decides where
/// the bytes came from, which keeps downloading and caching out of this crate
/// and lets the app fetch weights on demand.
pub struct BurnEmbedder {
    name: String,
}

impl BurnEmbedder {
    pub fn from_bytes(_weights: &[u8]) -> Result<Self> {
        if !HAS_MODEL {
            bail!(
                "this build has no model compiled in — run scripts/fetch-model.sh, \
                 then rebuild"
            );
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
