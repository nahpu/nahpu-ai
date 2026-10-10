use anyhow::{bail, Result};
use nahpu_embed_core::{ImageEmbedder, TextEmbedder};

pub const CLIP_MODEL_NAMES: [&str; 1] = ["jina-clip-v1"];

/// A text and an image tower trained into one shared space.
pub struct ClipModel {
    pub name: &'static str,
    pub text: Box<dyn TextEmbedder>,
    pub vision: Box<dyn ImageEmbedder>,
    pub logit_scale: f32,
}

pub fn load_clip(name: &str) -> Result<ClipModel> {
    match name {
        "jina-clip-v1" => Ok(ClipModel {
            name: "jina-clip-v1",
            text: crate::load_jina_text()?,
            vision: crate::load_jina_vision()?,
            logit_scale: crate::JINA_LOGIT_SCALE,
        }),
        other => bail!("unknown model {other:?}; known: {}", CLIP_MODEL_NAMES.join(", ")),
    }
}
