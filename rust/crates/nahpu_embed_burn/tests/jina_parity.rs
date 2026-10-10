use nahpu_embed_burn::{load_jina_text, load_jina_vision, models_dir, HAS_JINA};
use nahpu_embed_core::preprocess::{decode_upright, normalise_to_chw, resize_and_centre_crop};
use nahpu_embed_core::{cosine_similarity, TextRole};

const TEXT_REFERENCE: &str = include_str!("data/jina-text-reference.tsv");
const IMAGE_REFERENCE: &str = include_str!("data/jina-image-reference.tsv");

/// Same tensor in, so only Burn can differ from onnxruntime.
const MODEL_GATE: f32 = 0.9999;
/// Our decode and resize may differ from PIL by a few levels per pixel.
const END_TO_END_GATE: f32 = 0.998;
/// zune-jpeg vs libjpeg and fixed-point rounding give at most 3 levels, mean
/// about 0.1 at worst. The smallest real mistake measured (bilinear instead of bicubic)
/// gives mean 0.42 and max 9; a 1 px crop shift gives mean 5 or more.
const MAX_PIXEL_DIFFERENCE: u8 = 4;
const MAX_MEAN_PIXEL_DIFFERENCE: f64 = 0.15;

#[test]
fn text_matches_onnxruntime() {
    if skip_without_jina() {
        return;
    }
    let embedder = load_jina_text().unwrap();

    for (prompt, expected) in reference_rows(TEXT_REFERENCE) {
        let actual = embedder.embed(&prompt, TextRole::Query).unwrap();
        let cosine = cosine_similarity(&actual, &expected);
        assert!(cosine > MODEL_GATE, "{prompt:?}: cosine {cosine}");
    }
}

#[test]
fn padding_in_a_batch_does_not_change_a_text_vector() {
    if skip_without_jina() {
        return;
    }
    let embedder = load_jina_text().unwrap();
    let prompts: Vec<String> = reference_rows(TEXT_REFERENCE).into_iter().map(|(p, _)| p).collect();

    let batched = embedder.embed_batch(&prompts, TextRole::Query).unwrap();
    for (prompt, batch_vector) in prompts.iter().zip(&batched) {
        let alone = embedder.embed(prompt, TextRole::Query).unwrap();
        let cosine = cosine_similarity(&alone, batch_vector);
        assert!(cosine > MODEL_GATE, "{prompt:?}: cosine {cosine}");
    }
}

#[test]
fn vision_matches_onnxruntime_on_the_same_pixels() {
    if skip_without_jina() {
        return;
    }
    let embedder = load_jina_vision().unwrap();

    for (fixture, expected) in reference_rows(IMAGE_REFERENCE) {
        let pil_crop = reference_crop(&fixture);
        let pixels = normalise_to_chw(&pil_crop, &embedder.preprocess_config());

        let actual = embedder.embed_pixels(&pixels).unwrap();
        let cosine = cosine_similarity(&actual, &expected);
        assert!(cosine > MODEL_GATE, "{fixture}: cosine {cosine}");
    }
}

#[test]
fn our_crop_matches_pil_up_to_rounding() {
    if skip_without_jina() {
        return;
    }
    for (fixture, _) in reference_rows(IMAGE_REFERENCE) {
        let photo = decode_upright(&read_fixture_photo(&fixture)).unwrap();
        let ours = resize_and_centre_crop(&photo, 224).unwrap();
        let pil = reference_crop(&fixture);

        let differences: Vec<u8> = ours.as_raw().iter().zip(pil.as_raw()).map(|(a, b)| a.abs_diff(*b)).collect();
        let largest = *differences.iter().max().unwrap();
        let mean = differences.iter().map(|&d| f64::from(d)).sum::<f64>() / differences.len() as f64;
        eprintln!("{fixture}: largest difference {largest}, mean {mean:.4}");
        assert!(largest <= MAX_PIXEL_DIFFERENCE, "{fixture}: largest difference {largest}");
        assert!(mean < MAX_MEAN_PIXEL_DIFFERENCE, "{fixture}: mean difference {mean}");
    }
}

#[test]
fn vision_matches_onnxruntime_from_the_photo_file() {
    if skip_without_jina() {
        return;
    }
    let embedder = load_jina_vision().unwrap();

    for (fixture, expected) in reference_rows(IMAGE_REFERENCE) {
        let actual = embedder.embed_image(&read_fixture_photo(&fixture)).unwrap();
        let cosine = cosine_similarity(&actual, &expected);
        eprintln!("{fixture}: end-to-end cosine {cosine:.6}");
        assert!(cosine > END_TO_END_GATE, "{fixture}: cosine {cosine}");
    }
}

#[test]
fn reference_data_is_actually_loaded() {
    let text = reference_rows(TEXT_REFERENCE);
    let images = reference_rows(IMAGE_REFERENCE);
    assert_eq!(text.len(), 3);
    assert_eq!(images.len(), 2);
    assert!(text.iter().chain(&images).all(|(_, v)| v.len() == 768));
}

fn skip_without_jina() -> bool {
    let fixtures_present = reference_rows(IMAGE_REFERENCE)
        .iter()
        .all(|(fixture, _)| fixture_photo_path(fixture).exists());
    if HAS_JINA && fixtures_present {
        return false;
    }
    eprintln!("skipped: jina or its fixtures are missing — run scripts/fetch-model.sh, then rebuild");
    true
}

fn fixture_photo_path(fixture: &str) -> std::path::PathBuf {
    models_dir().join(format!("{fixture}.jpg"))
}

fn read_fixture_photo(fixture: &str) -> Vec<u8> {
    std::fs::read(fixture_photo_path(fixture)).unwrap()
}

fn reference_crop(fixture: &str) -> image::RgbImage {
    let bytes = std::fs::read(format!("{}/tests/data/{fixture}.pixels.rgb", env!("CARGO_MANIFEST_DIR"))).unwrap();
    image::RgbImage::from_raw(224, 224, bytes).unwrap()
}

/// The vector is always the last tab-separated column.
fn reference_rows(table: &str) -> Vec<(String, Vec<f32>)> {
    table
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let columns: Vec<&str> = line.split('\t').collect();
            let vector = columns.last().unwrap().split(',').map(|x| x.parse().unwrap()).collect();
            (columns[0].to_string(), vector)
        })
        .collect()
}
