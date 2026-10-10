mod common;

use common::{fixture_photos_present, read_fixture_photo, reference_crop, reference_rows};
use nahpu_embed_burn::{load_bioclip_text, load_bioclip_vision, HAS_BIOCLIP};
use nahpu_embed_core::preprocess::normalise_to_chw;
use nahpu_embed_core::{cosine_similarity, TextRole};

const TEXT_REFERENCE: &str = include_str!("data/bioclip-text-reference.tsv");
const IMAGE_REFERENCE: &str = include_str!("data/bioclip-image-reference.tsv");
const PHOTO_REFERENCE: &str = include_str!("data/bioclip-photo-reference.tsv");

/// References come from open_clip in PyTorch, so these gates also cover the
/// ONNX export and its shape folding.
const MODEL_GATE: f32 = 0.9999;
const END_TO_END_GATE: f32 = 0.998;

#[test]
fn text_matches_open_clip() {
    if skip_without_bioclip() {
        return;
    }
    let embedder = load_bioclip_text().unwrap();

    for (prompt, expected) in reference_rows(TEXT_REFERENCE) {
        let actual = embedder.embed(&prompt, TextRole::Query).unwrap();
        let cosine = cosine_similarity(&actual, &expected);
        eprintln!("{prompt:?}: cosine {cosine:.7}");
        assert!(cosine > MODEL_GATE, "{prompt:?}: cosine {cosine}");
    }
}

#[test]
fn vision_matches_open_clip_on_the_same_pixels() {
    if skip_without_bioclip() {
        return;
    }
    let embedder = load_bioclip_vision().unwrap();

    for (fixture, expected) in reference_rows(IMAGE_REFERENCE) {
        let pixels = normalise_to_chw(&reference_crop(&fixture), &embedder.preprocess_config());
        let actual = embedder.embed_pixels(&pixels).unwrap();
        let cosine = cosine_similarity(&actual, &expected);
        eprintln!("{fixture}: cosine {cosine:.7}");
        assert!(cosine > MODEL_GATE, "{fixture}: cosine {cosine}");
    }
}

#[test]
fn vision_matches_open_clip_preprocessing_from_the_photo_file() {
    if skip_without_bioclip() {
        return;
    }
    let embedder = load_bioclip_vision().unwrap();

    for (fixture, expected) in reference_rows(PHOTO_REFERENCE) {
        let actual = embedder.embed_image(&read_fixture_photo(&fixture)).unwrap();
        let cosine = cosine_similarity(&actual, &expected);
        eprintln!("{fixture}: end-to-end cosine {cosine:.6}");
        assert!(cosine > END_TO_END_GATE, "{fixture}: cosine {cosine}");
    }
}

#[test]
fn reference_data_is_actually_loaded() {
    assert_eq!(reference_rows(TEXT_REFERENCE).len(), 3);
    for table in [IMAGE_REFERENCE, PHOTO_REFERENCE] {
        let rows = reference_rows(table);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|(_, v)| v.len() == 512));
    }
}

fn skip_without_bioclip() -> bool {
    if HAS_BIOCLIP && fixture_photos_present(IMAGE_REFERENCE) {
        return false;
    }
    eprintln!("skipped: BioCLIP or the fixtures are missing — run scripts/bioclip/export_onnx.py and scripts/fetch-model.sh, then rebuild");
    true
}
