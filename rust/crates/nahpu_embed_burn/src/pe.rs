//! Meta's PE-Core (Perception Encoder): a text and an image tower in one
//! space, in five sizes. Each is exported by `scripts/pe/export_onnx.py`.

use anyhow::Result;
use nahpu_embed_core::{ImageEmbedder, TextEmbedder};

/// One size of PE-Core.
pub struct PeCoreSize {
    pub name: &'static str,
    /// exp of the trained `logit_scale` in the size's checkpoint.
    pub logit_scale: f32,
    /// Whether both towers are compiled in.
    pub compiled_in: bool,
    pub load_text: fn() -> Result<Box<dyn TextEmbedder>>,
    pub load_vision: fn() -> Result<Box<dyn ImageEmbedder>>,
}

pub const PE_CORE_SIZES: [PeCoreSize; 5] = [t16_384::SIZE, s16_384::SIZE, b16_224::SIZE, l14_336::SIZE, g14_448::SIZE];

/// Every size has its own generated model type, so the loading is written
/// once per size. What the towers do with a loaded model is in `towers`.
macro_rules! pe_core_size {
    (
        mod $module:ident,
        name: $name:literal,
        image_size: $image_size:literal,
        dimensions: $dimensions:literal,
        context_length: $context_length:literal,
        logit_scale: $logit_scale:literal,
        text: $text_cfg:ident $text_weights:literal,
        vision: $vision_cfg:ident $vision_weights:literal
    ) => {
        mod $module {
            use super::*;

            pub const SIZE: PeCoreSize = PeCoreSize {
                name: $name,
                logit_scale: $logit_scale,
                compiled_in: cfg!(all($text_cfg, $vision_cfg)),
                load_text,
                load_vision,
            };

            fn load_text() -> Result<Box<dyn TextEmbedder>> {
                #[cfg($text_cfg)]
                return Ok(Box::new(text::load()?));
                #[cfg(not($text_cfg))]
                return Err(missing($name, "text"));
            }

            fn load_vision() -> Result<Box<dyn ImageEmbedder>> {
                #[cfg($vision_cfg)]
                return Ok(Box::new(vision::load()?));
                #[cfg(not($vision_cfg))]
                return Err(missing($name, "vision"));
            }

            #[cfg($text_cfg)]
            mod text {
                use super::super::towers::TextTower;

                #[allow(clippy::all, clippy::pedantic, dead_code)]
                mod model {
                    include!(concat!(env!("OUT_DIR"), "/model/", $name, "-text.rs"));
                }

                pub fn load() -> anyhow::Result<TextTower> {
                    let device = crate::selected_device();
                    let weights = crate::read_weights(env!($text_weights))?;
                    let model = model::Model::from_bytes(weights, &device);

                    Ok(TextTower {
                        forward: Box::new(move |input_ids| model.forward(input_ids)),
                        tokenizer: crate::read_tokenizer($name)?,
                        device,
                        name: format!(concat!("{}/", $name, "-text"), crate::BACKEND_NAME),
                        dimensions: $dimensions,
                        context_length: $context_length,
                    })
                }
            }

            #[cfg($vision_cfg)]
            mod vision {
                use super::super::towers::VisionTower;

                #[allow(clippy::all, clippy::pedantic, dead_code)]
                mod model {
                    include!(concat!(env!("OUT_DIR"), "/model/", $name, "-vision.rs"));
                }

                pub fn load() -> anyhow::Result<VisionTower> {
                    let device = crate::selected_device();
                    let weights = crate::read_weights(env!($vision_weights))?;
                    let model = model::Model::from_bytes(weights, &device);

                    Ok(VisionTower {
                        forward: Box::new(move |pixel_values| model.forward(pixel_values)),
                        device,
                        name: format!(concat!("{}/", $name, "-vision"), crate::BACKEND_NAME),
                        dimensions: $dimensions,
                        image_size: $image_size,
                    })
                }
            }
        }
    };
}

pe_core_size! {
    mod t16_384,
    name: "pe-core-t16-384",
    image_size: 384,
    dimensions: 512,
    context_length: 32,
    logit_scale: 100.0,
    text: has_pe_core_t_text "PE_CORE_T_TEXT_WEIGHTS",
    vision: has_pe_core_t_vision "PE_CORE_T_VISION_WEIGHTS"
}

pe_core_size! {
    mod s16_384,
    name: "pe-core-s16-384",
    image_size: 384,
    dimensions: 512,
    context_length: 32,
    logit_scale: 100.0,
    text: has_pe_core_s_text "PE_CORE_S_TEXT_WEIGHTS",
    vision: has_pe_core_s_vision "PE_CORE_S_VISION_WEIGHTS"
}

pe_core_size! {
    mod b16_224,
    name: "pe-core-b16-224",
    image_size: 224,
    dimensions: 1024,
    context_length: 32,
    logit_scale: 99.93,
    text: has_pe_core_b_text "PE_CORE_B_TEXT_WEIGHTS",
    vision: has_pe_core_b_vision "PE_CORE_B_VISION_WEIGHTS"
}

pe_core_size! {
    mod l14_336,
    name: "pe-core-l14-336",
    image_size: 336,
    dimensions: 1024,
    context_length: 32,
    logit_scale: 99.89,
    text: has_pe_core_l_text "PE_CORE_L_TEXT_WEIGHTS",
    vision: has_pe_core_l_vision "PE_CORE_L_VISION_WEIGHTS"
}

// Exported in half precision: in single precision its image tower is 7.6 GB,
// more than an 8 GB card holds.
pe_core_size! {
    mod g14_448,
    name: "pe-core-g14-448",
    image_size: 448,
    dimensions: 1280,
    context_length: 72,
    logit_scale: 99.83,
    text: has_pe_core_g_text "PE_CORE_G_TEXT_WEIGHTS",
    vision: has_pe_core_g_vision "PE_CORE_G_VISION_WEIGHTS"
}

// Unused when every size is compiled in.
#[allow(dead_code)]
fn missing(name: &str, tower: &str) -> anyhow::Error {
    anyhow::anyhow!("{name}-{tower} is not compiled in — run scripts/pe/export_onnx.py {name}, then rebuild")
}

// Unused when no size is compiled in.
#[allow(dead_code, unused_imports)]
mod towers {
    use anyhow::Result;
    use burn::prelude::*;
    use nahpu_embed_core::{
        normalise_to_unit_length, Embedding, ImageEmbedder, PreprocessConfig, TextEmbedder, TextRole,
    };
    use tokenizers::Tokenizer;

    const END_OF_TEXT: u32 = 49407;

    pub struct TextTower {
        pub forward: Box<dyn Fn(Tensor<2, Int>) -> Tensor<2>>,
        pub tokenizer: Tokenizer,
        pub device: Device,
        pub name: String,
        pub dimensions: usize,
        /// The tokens the size was trained on: 32, or 72 for G.
        pub context_length: usize,
    }

    impl TextTower {
        fn embed_one(&self, text: &str) -> Result<Embedding> {
            let encoded = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("could not tokenize: {e}"))?;
            let ids = fit_to_context(encoded.get_ids(), self.context_length);
            let input_ids =
                Tensor::<2, Int>::from_data(TensorData::new(ids, [1, self.context_length]), &self.device);

            let output = (self.forward)(input_ids);
            Ok(normalise_to_unit_length(crate::rows_of(output).remove(0)))
        }
    }

    /// Mirrors open_clip's tokenizer, as BioCLIP: cut to the context keeping
    /// end-of-text last, pad with 0.
    fn fit_to_context(ids: &[u32], context_length: usize) -> Vec<i64> {
        let mut fitted: Vec<i64> = ids.iter().take(context_length).map(|&id| i64::from(id)).collect();
        if ids.len() > context_length {
            fitted[context_length - 1] = i64::from(END_OF_TEXT);
        }
        fitted.resize(context_length, 0);
        fitted
    }

    impl TextEmbedder for TextTower {
        fn name(&self) -> &str {
            &self.name
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn embed(&self, text: &str, _role: TextRole) -> Result<Embedding> {
            self.embed_one(text)
        }

        /// The export is fixed at batch 1, so prompts go through one at a time.
        fn embed_batch(&self, texts: &[String], _role: TextRole) -> Result<Vec<Embedding>> {
            texts.iter().map(|text| self.embed_one(text)).collect()
        }
    }

    pub struct VisionTower {
        pub forward: Box<dyn Fn(Tensor<4>) -> Tensor<2>>,
        pub device: Device,
        pub name: String,
        pub dimensions: usize,
        pub image_size: u32,
    }

    impl ImageEmbedder for VisionTower {
        fn name(&self) -> &str {
            &self.name
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        /// The model's own config also squashes the photo to a square with a
        /// bilinear filter. Here the photo is framed as for every other model.
        fn preprocess_config(&self) -> PreprocessConfig {
            PreprocessConfig {
                size: self.image_size,
                mean: [0.5; 3],
                std: [0.5; 3],
            }
        }

        fn embed_pixels(&self, pixels: &[f32]) -> Result<Embedding> {
            let size = self.image_size as usize;
            anyhow::ensure!(
                pixels.len() == 3 * size * size,
                "expected a 3x{size}x{size} tensor, got {} values",
                pixels.len()
            );
            let input = Tensor::<4>::from_data(TensorData::new(pixels.to_vec(), [1, 3, size, size]), &self.device);

            let output = (self.forward)(input);
            Ok(normalise_to_unit_length(crate::rows_of(output).remove(0)))
        }
    }
}
