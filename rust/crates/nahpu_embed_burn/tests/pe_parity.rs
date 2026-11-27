mod common;

use common::{fixture_photos_present, read_fixture_photo, reference_crop, reference_rows};
use nahpu_embed_burn::{PeCoreSize, PE_CORE_SIZES};
use nahpu_embed_core::preprocess::normalise_to_chw;
use nahpu_embed_core::{cosine_similarity, TextRole};

/// What `scripts/pe/export_onnx.py` wrote for one size.
struct References {
    name: &'static str,
    dimensions: usize,
    /// The lowest cosine against PyTorch on the same token ids or pixels.
    model_gate: f32,
    text: &'static str,
    image: &'static str,
    photo: &'static str,
}

macro_rules! references {
    ($name:literal, $dimensions:literal, $letter:literal, $model_gate:ident) => {
        References {
            name: $name,
            dimensions: $dimensions,
            model_gate: $model_gate,
            text: include_str!(concat!("data/pe-core-", $letter, "-text-reference.tsv")),
            image: include_str!(concat!("data/pe-core-", $letter, "-image-reference.tsv")),
            photo: include_str!(concat!("data/pe-core-", $letter, "-photo-reference.tsv")),
        }
    };
}

const REFERENCES: [References; 5] = [
    references!("pe-core-t16-384", 512, "t", MODEL_GATE),
    references!("pe-core-s16-384", 512, "s", MODEL_GATE),
    references!("pe-core-b16-224", 1024, "b", MODEL_GATE),
    references!("pe-core-l14-336", 1024, "l", MODEL_GATE),
    references!("pe-core-g14-448", 1280, "g", HALF_PRECISION_GATE),
];

/// References come from open_clip in PyTorch, so these gates also cover the
/// ONNX export and its shape folding.
const MODEL_GATE: f32 = 0.9999;
/// G is exported in half precision and its references are in single
/// precision. Its image tower gets 0.9998 to 0.9999 on the same pixels;
/// PyTorch's own half precision gets 0.99984 against its single precision.
const HALF_PRECISION_GATE: f32 = 0.9995;
/// The reference resizes with a bicubic filter, as this crate does for every
/// model. The model's own config asks for a bilinear one, which is a different
/// vector: cosine 0.987 to 0.989 for L on these fixtures, measured in PyTorch.
const END_TO_END_GATE: f32 = 0.998;

#[test]
fn text_matches_open_clip() {
    for (size, references) in sizes_to_test() {
        let embedder = (size.load_text)().unwrap();

        for (prompt, expected) in reference_rows(references.text) {
            let actual = embedder.embed(&prompt, TextRole::Query).unwrap();
            let cosine = cosine_similarity(&actual, &expected);
            eprintln!("{} {prompt:?}: cosine {cosine:.7}", size.name);
            assert!(cosine > references.model_gate, "{} {prompt:?}: cosine {cosine}", size.name);
        }
    }
}

/// Each tower is loaded once: G's image tower is 3.8 GB, and loading it a
/// second time in one process runs an 8 GB card out of memory.
#[test]
fn vision_matches_open_clip_on_the_same_pixels_and_from_the_photo_file() {
    for (size, references) in sizes_to_test() {
        let embedder = (size.load_vision)().unwrap();
        let config = embedder.preprocess_config();

        for (fixture, expected) in reference_rows(references.image) {
            let pixels = normalise_to_chw(&on_grey_canvas(&reference_crop(&fixture), config.size), &config);
            let actual = embedder.embed_pixels(&pixels).unwrap();
            let cosine = cosine_similarity(&actual, &expected);
            eprintln!("{} {fixture}: cosine {cosine:.7}", size.name);
            assert!(cosine > references.model_gate, "{} {fixture}: cosine {cosine}", size.name);
        }

        for (fixture, expected) in reference_rows(references.photo) {
            let actual = embedder.embed_image(&read_fixture_photo(&fixture)).unwrap();
            let cosine = cosine_similarity(&actual, &expected);
            eprintln!("{} {fixture}: end-to-end cosine {cosine:.6}", size.name);
            assert!(cosine > END_TO_END_GATE, "{} {fixture}: cosine {cosine}", size.name);
        }
    }
}

#[test]
fn reference_data_is_actually_loaded() {
    for references in &REFERENCES {
        assert_eq!(reference_rows(references.text).len(), 3, "{}", references.name);
        for table in [references.image, references.photo] {
            let rows = reference_rows(table);
            assert_eq!(rows.len(), 2, "{}", references.name);
            assert!(rows.iter().all(|(_, vector)| vector.len() == references.dimensions), "{}", references.name);
        }
    }
}

/// The 224 fixture crop in the middle of a mid-grey square of the model's
/// size, as scripts/pe/export_onnx.py builds it: the same pixels on both sides.
fn on_grey_canvas(crop: &image::RgbImage, size: u32) -> image::RgbImage {
    let mut canvas = image::RgbImage::from_pixel(size, size, image::Rgb([128; 3]));
    let offset = i64::from((size - crop.width()) / 2);
    image::imageops::replace(&mut canvas, crop, offset, offset);
    canvas
}

/// The sizes that are compiled in, each with its references.
fn sizes_to_test() -> Vec<(&'static PeCoreSize, &'static References)> {
    if !fixture_photos_present(REFERENCES[0].image) {
        eprintln!("skipped: the fixture photos are missing — run scripts/fetch-model.sh");
        return Vec::new();
    }

    let mut compiled_in = Vec::new();
    for references in &REFERENCES {
        let size = PE_CORE_SIZES.iter().find(|size| size.name == references.name).expect("a reference for every size");
        if size.compiled_in {
            compiled_in.push((size, references));
        } else {
            eprintln!("skipped {}: not compiled in — run scripts/pe/export_onnx.py {0}, then rebuild", size.name);
        }
    }
    compiled_in
}
