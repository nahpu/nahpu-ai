//! Saves image vectors for a manifest, plus the label vectors for one setting,
//! as `.npy` files that numpy reads directly.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;

use nahpu_embed_burn::{load_clip, BACKEND_NAME};
use nahpu_embed_core::preprocess::Framing;
use nahpu_embed_core::zero_shot::{read_labels_csv, LabelWording, PromptSet, ZeroShotClassifier};
use nahpu_embed_core::Embedding;

use crate::eval::wording_name;
use crate::photos::{embed_all, framing_name, read_manifest};

pub struct EmbedPlan {
    pub model: String,
    pub manifest: PathBuf,
    pub labels: PathBuf,
    pub out: PathBuf,
    pub framing: Framing,
    pub prompts: PromptSet,
    pub wording: LabelWording,
}

#[derive(Serialize)]
struct Meta {
    model: String,
    backend: &'static str,
    manifest: String,
    framing: &'static str,
    prompts: &'static str,
    wording: &'static str,
    images: usize,
    dimensions: usize,
    git_commit: String,
}

pub fn run(plan: &EmbedPlan) -> Result<()> {
    let labels = read_labels_csv(&std::fs::read_to_string(&plan.labels).context("could not read the labels file")?)?;
    let photos = read_manifest(&plan.manifest)?;
    let model = load_clip(&plan.model)?;
    let classifier = ZeroShotClassifier::new(model.text.as_ref(), &labels, plan.wording, plan.prompts, model.logit_scale)?;

    let (vectors, _) = embed_all(&model, &photos, plan.framing)?;

    std::fs::create_dir_all(&plan.out)?;
    write_npy(&plan.out.join("images.npy"), &vectors)?;
    let mut rows = csv::Writer::from_path(plan.out.join("images.csv"))?;
    rows.write_record(["row", "path", "class"])?;
    for (row, photo) in photos.iter().enumerate() {
        rows.write_record([row.to_string().as_str(), &photo.listed_as, &photo.class])?;
    }
    rows.flush()?;

    write_npy(&plan.out.join("labels.npy"), classifier.class_vectors())?;
    let mut rows = csv::Writer::from_path(plan.out.join("labels.csv"))?;
    rows.write_record(["row", "class"])?;
    for (row, class) in classifier.classes().iter().enumerate() {
        rows.write_record([row.to_string().as_str(), class])?;
    }
    rows.flush()?;

    let meta = Meta {
        model: plan.model.clone(),
        backend: BACKEND_NAME,
        manifest: plan.manifest.display().to_string(),
        framing: framing_name(plan.framing),
        prompts: plan.prompts.name(),
        wording: wording_name(plan.wording),
        images: vectors.len(),
        dimensions: vectors.first().map_or(0, Vec::len),
        git_commit: git_commit(),
    };
    std::fs::write(plan.out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    eprintln!("wrote {}", plan.out.display());
    Ok(())
}

/// Writes a 2D little-endian float32 array in `.npy` format version 1.0.
fn write_npy(path: &Path, vectors: &[Embedding]) -> Result<()> {
    let width = vectors.first().map_or(0, Vec::len);
    if vectors.iter().any(|v| v.len() != width) {
        bail!("vectors for {} have different lengths", path.display());
    }

    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {width}), }}", vectors.len());
    // The magic string, version and length take 10 bytes; the header ends in a newline at a 64-byte boundary.
    let padding = 64 - (10 + header.len() + 1) % 64;
    header.push_str(&" ".repeat(padding % 64));
    header.push('\n');

    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&u16::try_from(header.len())?.to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    for value in vectors.iter().flatten() {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map_or_else(|| "unknown".into(), |output| String::from_utf8_lossy(&output.stdout).trim().to_string())
}
