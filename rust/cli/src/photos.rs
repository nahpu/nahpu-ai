//! Reads a labelled photo manifest and embeds every photo in it.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};

use nahpu_embed_burn::ClipModel;
use nahpu_embed_core::preprocess::{preprocess_framed, Framing};
use nahpu_embed_core::Embedding;

pub struct Photo {
    pub path: PathBuf,
    pub listed_as: String,
    pub class: String,
}

pub fn read_manifest(manifest: &Path) -> Result<Vec<Photo>> {
    let base = manifest.parent().unwrap_or(Path::new("."));
    let mut reader = csv::Reader::from_path(manifest).with_context(|| format!("could not read {}", manifest.display()))?;
    let headers = reader.headers()?.clone();
    let column = |name: &str| headers.iter().position(|h| h == name).with_context(|| format!("manifest has no {name} column"));
    let (path_column, class_column) = (column("path")?, column("class")?);

    reader
        .records()
        .map(|record| {
            let record = record?;
            Ok(Photo {
                path: base.join(&record[path_column]),
                listed_as: record[path_column].to_string(),
                class: record[class_column].to_string(),
            })
        })
        .collect()
}

/// Returns one vector per photo, in manifest order, and the milliseconds each took.
pub fn embed_all(model: &ClipModel, photos: &[Photo], framing: Framing) -> Result<(Vec<Embedding>, Vec<f64>)> {
    let config = model.vision.preprocess_config();
    let mut vectors = Vec::with_capacity(photos.len());
    let mut milliseconds = Vec::with_capacity(photos.len());

    for (done, photo) in photos.iter().enumerate() {
        let started = Instant::now();
        let encoded = std::fs::read(&photo.path).with_context(|| format!("could not read {}", photo.path.display()))?;
        let pixels = preprocess_framed(&encoded, &config, framing)?;
        vectors.push(model.vision.embed_pixels(&pixels)?);
        milliseconds.push(started.elapsed().as_secs_f64() * 1000.0);

        if (done + 1) % 100 == 0 {
            eprintln!("  {} {}/{}", framing_name(framing), done + 1, photos.len());
        }
    }
    Ok((vectors, milliseconds))
}

pub fn framing_name(framing: Framing) -> &'static str {
    match framing {
        Framing::CentreCrop => "centre-crop",
        Framing::PadToSquare => "pad",
    }
}
