#![allow(dead_code)]

use nahpu_embed_burn::models_dir;

pub fn fixture_photos_present(image_reference: &str) -> bool {
    reference_rows(image_reference)
        .iter()
        .all(|(fixture, _)| fixture_photo_path(fixture).exists())
}

pub fn fixture_photo_path(fixture: &str) -> std::path::PathBuf {
    models_dir().join(format!("{fixture}.jpg"))
}

pub fn read_fixture_photo(fixture: &str) -> Vec<u8> {
    std::fs::read(fixture_photo_path(fixture)).unwrap()
}

pub fn reference_crop(fixture: &str) -> image::RgbImage {
    let bytes = std::fs::read(format!("{}/tests/data/{fixture}.pixels.rgb", env!("CARGO_MANIFEST_DIR"))).unwrap();
    image::RgbImage::from_raw(224, 224, bytes).unwrap()
}

/// The vector is always the last tab-separated column.
pub fn reference_rows(table: &str) -> Vec<(String, Vec<f32>)> {
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
