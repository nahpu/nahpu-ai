//! Runs zero-shot classification over a labelled manifest and writes
//! predictions, metrics and a Markdown report.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::Serialize;

use nahpu_embed_burn::{load_clip, ClipModel, BACKEND_NAME};
use nahpu_embed_core::preprocess::{preprocess_framed, Framing};
use nahpu_embed_core::zero_shot::{read_labels_csv, Label, LabelWording, PromptSet, ZeroShotClassifier};
use nahpu_embed_core::Embedding;

pub struct EvalPlan {
    pub model: String,
    pub manifest: PathBuf,
    pub labels: PathBuf,
    pub out: PathBuf,
    pub framings: Vec<Framing>,
    pub settings: Vec<(PromptSet, LabelWording)>,
    pub limit: Option<usize>,
}

struct Photo {
    path: PathBuf,
    listed_as: String,
    class: String,
}

#[derive(Serialize)]
struct Summary {
    model: String,
    backend: &'static str,
    manifest: String,
    images: usize,
    images_per_class: BTreeMap<String, usize>,
    load_seconds: f64,
    image_ms_median: f64,
    image_ms_p95: f64,
    classes: Vec<String>,
    runs: Vec<RunMetrics>,
}

#[derive(Serialize)]
struct RunMetrics {
    framing: &'static str,
    prompts: &'static str,
    wording: &'static str,
    top1: f64,
    top3: f64,
    macro_accuracy: f64,
    per_class_accuracy: BTreeMap<String, f64>,
    /// Rows are the true class, columns the predicted class, both in `classes` order.
    confusion: Vec<Vec<usize>>,
}

pub fn run(plan: &EvalPlan) -> Result<()> {
    let labels = read_labels_csv(&std::fs::read_to_string(&plan.labels).context("could not read the labels file")?)?;
    let mut photos = read_manifest(&plan.manifest)?;
    if let Some(limit) = plan.limit {
        photos.truncate(limit);
    }
    check_every_class_has_a_label(&photos, &labels)?;

    let started = Instant::now();
    let model = load_clip(&plan.model)?;
    let load_seconds = started.elapsed().as_secs_f64();

    let classifiers = plan
        .settings
        .iter()
        .map(|&(prompts, wording)| {
            let classifier =
                ZeroShotClassifier::new(model.text.as_ref(), &labels, wording, prompts, model.logit_scale)?;
            Ok((prompts, wording, classifier))
        })
        .collect::<Result<Vec<_>>>()?;

    std::fs::create_dir_all(plan.out.join("predictions"))?;
    let mut runs = Vec::new();
    let mut timings = Vec::new();
    for &framing in &plan.framings {
        let (vectors, milliseconds) = embed_all(&model, &photos, framing)?;
        if timings.is_empty() {
            timings = milliseconds;
        }
        for (prompts, wording, classifier) in &classifiers {
            let metrics = score(&photos, &vectors, classifier, framing, *prompts, *wording, &plan.out)?;
            runs.push(metrics);
        }
    }

    let summary = Summary {
        model: plan.model.clone(),
        backend: BACKEND_NAME,
        manifest: plan.manifest.display().to_string(),
        images: photos.len(),
        images_per_class: count_per_class(&photos),
        load_seconds,
        image_ms_median: percentile(&timings, 0.5),
        image_ms_p95: percentile(&timings, 0.95),
        classes: labels.iter().map(|l| l.class.clone()).collect(),
        runs,
    };
    std::fs::write(plan.out.join("metrics.json"), serde_json::to_string_pretty(&summary)?)?;
    std::fs::write(plan.out.join("report.md"), report(&summary))?;
    eprintln!("wrote {}", plan.out.display());
    Ok(())
}

fn read_manifest(manifest: &Path) -> Result<Vec<Photo>> {
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

fn check_every_class_has_a_label(photos: &[Photo], labels: &[Label]) -> Result<()> {
    for photo in photos {
        if !labels.iter().any(|label| label.class == photo.class) {
            bail!("{} has class {:?}, which the labels file does not list", photo.listed_as, photo.class);
        }
    }
    Ok(())
}

fn embed_all(model: &ClipModel, photos: &[Photo], framing: Framing) -> Result<(Vec<Embedding>, Vec<f64>)> {
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

fn score(
    photos: &[Photo],
    vectors: &[Embedding],
    classifier: &ZeroShotClassifier,
    framing: Framing,
    prompts: PromptSet,
    wording: LabelWording,
    out: &Path,
) -> Result<RunMetrics> {
    let classes = classifier.classes();
    let index_of = |class: &str| classes.iter().position(|c| c == class).expect("checked against labels");

    let mut confusion = vec![vec![0; classes.len()]; classes.len()];
    let mut in_top3 = 0;
    let file = out
        .join("predictions")
        .join(format!("{}-{}-{}.csv", framing_name(framing), prompts.name(), wording_name(wording)));
    let mut writer = csv::Writer::from_path(&file)?;
    let mut header = vec!["path".to_string(), "truth".into(), "predicted".into(), "correct".into()];
    header.extend(classes.iter().map(|c| format!("cos:{c}")));
    writer.write_record(&header)?;

    for (photo, vector) in photos.iter().zip(vectors) {
        let ranked = classifier.rank(vector);
        let predicted = &ranked[0].class;
        confusion[index_of(&photo.class)][index_of(predicted)] += 1;
        if ranked.iter().take(3).any(|p| p.class == photo.class) {
            in_top3 += 1;
        }

        let mut row = vec![
            photo.listed_as.clone(),
            photo.class.clone(),
            predicted.clone(),
            (predicted == &photo.class).to_string(),
        ];
        let cosine_of = |class: &String| ranked.iter().find(|p| &p.class == class).map_or(0.0, |p| p.cosine);
        row.extend(classes.iter().map(|c| format!("{:.6}", cosine_of(c))));
        writer.write_record(&row)?;
    }
    writer.flush()?;

    let per_class_accuracy: BTreeMap<String, f64> = classes
        .iter()
        .enumerate()
        .filter(|(i, _)| confusion[*i].iter().sum::<usize>() > 0)
        .map(|(i, class)| (class.clone(), confusion[i][i] as f64 / confusion[i].iter().sum::<usize>() as f64))
        .collect();
    let correct: usize = (0..classes.len()).map(|i| confusion[i][i]).sum();

    Ok(RunMetrics {
        framing: framing_name(framing),
        prompts: prompts.name(),
        wording: wording_name(wording),
        top1: correct as f64 / photos.len() as f64,
        top3: f64::from(in_top3) / photos.len() as f64,
        macro_accuracy: per_class_accuracy.values().sum::<f64>() / per_class_accuracy.len() as f64,
        per_class_accuracy,
        confusion,
    })
}

fn report(summary: &Summary) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Zero-shot evaluation: {}\n", summary.model);
    let _ = writeln!(md, "- Backend: {}", summary.backend);
    let _ = writeln!(md, "- Manifest: `{}`, {} images", summary.manifest, summary.images);
    let _ = writeln!(
        md,
        "- Model load {:.2} s; per image (decode + preprocess + embed) median {:.0} ms, p95 {:.0} ms\n",
        summary.load_seconds, summary.image_ms_median, summary.image_ms_p95
    );

    let _ = writeln!(md, "## All settings, best macro accuracy first\n");
    let _ = writeln!(md, "| Framing | Prompts | Wording | Top-1 | Top-3 | Macro |");
    let _ = writeln!(md, "|---|---|---|---|---|---|");
    let mut ranked: Vec<&RunMetrics> = summary.runs.iter().collect();
    ranked.sort_by(|a, b| b.macro_accuracy.total_cmp(&a.macro_accuracy));
    for run in &ranked {
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} | {} |",
            run.framing,
            run.prompts,
            run.wording,
            percent(run.top1),
            percent(run.top3),
            percent(run.macro_accuracy)
        );
    }

    let default = summary
        .runs
        .iter()
        .find(|r| r.framing == "centre-crop" && r.prompts == "ensemble" && r.wording == "common");
    if let Some(run) = default {
        detail(&mut md, summary, run, "Default setting (centre-crop, ensemble, common)");
    }
    if let Some(best) = ranked.first() {
        if default.is_none_or(|d| !std::ptr::eq(d, *best)) {
            detail(&mut md, summary, best, "Best setting");
        }
    }
    md
}

fn detail(md: &mut String, summary: &Summary, run: &RunMetrics, title: &str) {
    let _ = writeln!(md, "\n## {title}: {} / {} / {}\n", run.framing, run.prompts, run.wording);
    let _ = writeln!(md, "| Class | Images | Accuracy |");
    let _ = writeln!(md, "|---|---|---|");
    for class in &summary.classes {
        if let Some(accuracy) = run.per_class_accuracy.get(class) {
            let _ = writeln!(md, "| {class} | {} | {} |", summary.images_per_class[class], percent(*accuracy));
        }
    }

    let _ = writeln!(md, "\nConfusion matrix (rows = truth, columns = predicted):\n");
    let _ = writeln!(md, "| | {} |", summary.classes.join(" | "));
    let _ = writeln!(md, "|---|{}", "---|".repeat(summary.classes.len()));
    for (class, row) in summary.classes.iter().zip(&run.confusion) {
        let cells: Vec<String> = row.iter().map(usize::to_string).collect();
        let _ = writeln!(md, "| **{class}** | {} |", cells.join(" | "));
    }
}

fn count_per_class(photos: &[Photo]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for photo in photos {
        *counts.entry(photo.class.clone()).or_insert(0) += 1;
    }
    counts
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

fn percent(fraction: f64) -> String {
    format!("{:.1}%", fraction * 100.0)
}

pub fn framing_name(framing: Framing) -> &'static str {
    match framing {
        Framing::CentreCrop => "centre-crop",
        Framing::PadToSquare => "pad",
    }
}

pub fn wording_name(wording: LabelWording) -> &'static str {
    match wording {
        LabelWording::Common => "common",
        LabelWording::Taxon => "taxon",
        LabelWording::Both => "both",
    }
}
