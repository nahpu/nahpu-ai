use anyhow::{bail, Result};
use nahpu_embed_core::{ImageEmbedder, TextEmbedder};

use crate::PE_CORE_SIZES;

pub const CLIP_MODEL_NAMES: [&str; 8] = [
    "jina-clip-v1",
    "bioclip",
    "nomic-embed-v1.5",
    "pe-core-t16-384",
    "pe-core-s16-384",
    "pe-core-b16-224",
    "pe-core-l14-336",
    "pe-core-g14-448",
];

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
        "bioclip" => Ok(ClipModel {
            name: "bioclip",
            text: crate::load_bioclip_text()?,
            vision: crate::load_bioclip_vision()?,
            logit_scale: crate::BIOCLIP_LOGIT_SCALE,
        }),
        "nomic-embed-v1.5" => Ok(ClipModel {
            name: "nomic-embed-v1.5",
            text: crate::load_nomic_text()?,
            vision: crate::load_nomic_vision()?,
            logit_scale: crate::NOMIC_LOGIT_SCALE,
        }),
        other => match PE_CORE_SIZES.iter().find(|size| size.name == other) {
            Some(size) => Ok(ClipModel {
                name: size.name,
                text: (size.load_text)()?,
                vision: (size.load_vision)()?,
                logit_scale: size.logit_scale,
            }),
            None => bail!("unknown model {other:?}; known: {}", CLIP_MODEL_NAMES.join(", ")),
        },
    }
}
