mod embed;
mod eval;
mod ocr_eval;
mod photos;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

use nahpu_embed_burn::{
    load_clip, load_nomic_text, load_nomic_vision, BACKEND_NAME, CLIP_MODEL_NAMES, HAS_BIOCLIP, HAS_JINA, HAS_MINILM,
    HAS_NOMIC, PE_CORE_SIZES,
};
use nahpu_embed_core::preprocess::Framing;
use nahpu_embed_core::zero_shot::{read_labels_csv, LabelWording, PromptSet, ZeroShotClassifier};
use nahpu_embed_core::{cosine_similarity, TextRole};
use nahpu_ocr_burn::{OcrEngine, HAS_OCR};

#[derive(Parser)]
#[command(about = "Try NAHPU embedding and OCR models by hand")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Rank labels for each photo, best first.
    Classify {
        #[command(flatten)]
        setup: ClassifierSetup,
        #[arg(long, default_value_t = 5)]
        top: usize,
        #[arg(required = true)]
        photos: Vec<PathBuf>,
    },
    /// Score a labelled manifest and write predictions, metrics.json and report.md.
    Eval {
        #[command(flatten)]
        setup: ClassifierSetup,
        /// CSV with `path` (relative to the manifest) and `class` columns.
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = FramingArg::CentreCrop)]
        framing: FramingArg,
        /// Every framing, prompt set and wording. Photos are embedded once per framing.
        #[arg(long)]
        sweep: bool,
        /// Only the first N photos, for a quick check.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Save image vectors for a manifest, and the label vectors for one setting, as .npy files.
    Embed {
        #[command(flatten)]
        setup: ClassifierSetup,
        /// CSV with `path` (relative to the manifest) and `class` columns.
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = FramingArg::CentreCrop)]
        framing: FramingArg,
    },
    /// Compare photos and text in Nomic's shared space: the cosine of every pair.
    Compare {
        /// An existing file is embedded as an image, anything else as a text query.
        /// A name with an image extension that is not a file is an error.
        #[arg(required = true, num_args = 2..)]
        items: Vec<String>,
    },
    /// Print the text lines in an image, in reading order.
    Ocr { image: PathBuf },
    /// Read label photos, compare the text with PaddleOCR's and write metrics.json and report.md.
    OcrEval {
        /// Folder of `*.reference.json` files from scripts/ocr/make_references.py.
        #[arg(long)]
        references: PathBuf,
        /// Folder holding the photos those files name.
        #[arg(long)]
        photos: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Args)]
struct ClassifierSetup {
    #[arg(long, default_value = "jina-clip-v1")]
    model: String,
    /// CSV with header class,common_name,taxon_name.
    #[arg(long)]
    labels: PathBuf,
    #[arg(long, value_enum, default_value_t = Prompts::Ensemble)]
    prompts: Prompts,
    #[arg(long, value_enum, default_value_t = Wording::Common)]
    wording: Wording,
}

#[derive(Clone, Copy, ValueEnum)]
enum Prompts {
    Bare,
    Photo,
    Specimen,
    Ensemble,
}

#[derive(Clone, Copy, ValueEnum)]
enum FramingArg {
    CentreCrop,
    Pad,
}

impl From<FramingArg> for Framing {
    fn from(framing: FramingArg) -> Self {
        match framing {
            FramingArg::CentreCrop => Framing::CentreCrop,
            FramingArg::Pad => Framing::PadToSquare,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Wording {
    Common,
    Taxon,
    Both,
}

impl From<Prompts> for PromptSet {
    fn from(prompts: Prompts) -> Self {
        match prompts {
            Prompts::Bare => PromptSet::Bare,
            Prompts::Photo => PromptSet::Photo,
            Prompts::Specimen => PromptSet::Specimen,
            Prompts::Ensemble => PromptSet::Ensemble,
        }
    }
}

impl From<Wording> for LabelWording {
    fn from(wording: Wording) -> Self {
        match wording {
            Wording::Common => LabelWording::Common,
            Wording::Taxon => LabelWording::Taxon,
            Wording::Both => LabelWording::Both,
        }
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        None => {
            report_what_is_available();
            Ok(())
        }
        Some(Command::Classify { setup, top, photos }) => classify(&setup, top, &photos),
        Some(Command::Eval { setup, manifest, out, framing, sweep, limit }) => {
            let (framings, settings) = if sweep {
                let wordings = [LabelWording::Common, LabelWording::Taxon, LabelWording::Both];
                let settings = PromptSet::ALL.iter().flat_map(|&p| wordings.map(|w| (p, w))).collect();
                (vec![Framing::CentreCrop, Framing::PadToSquare], settings)
            } else {
                (vec![framing.into()], vec![(setup.prompts.into(), setup.wording.into())])
            };
            eval::run(&eval::EvalPlan { model: setup.model, manifest, labels: setup.labels, out, framings, settings, limit })
        }
        Some(Command::Embed { setup, manifest, out, framing }) => embed::run(&embed::EmbedPlan {
            model: setup.model,
            manifest,
            labels: setup.labels,
            out,
            framing: framing.into(),
            prompts: setup.prompts.into(),
            wording: setup.wording.into(),
        }),
        Some(Command::Compare { items }) => compare_items(&items),
        Some(Command::Ocr { image }) => read_text(&image),
        Some(Command::OcrEval { references, photos, out }) => {
            ocr_eval::run(&ocr_eval::OcrEvalPlan { references, photos, out })
        }
    }
}

fn report_what_is_available() {
    let present = |yes: bool| if yes { "compiled in" } else { "missing — run scripts/fetch-model.sh, then rebuild" };
    println!("backend:       {BACKEND_NAME}");
    println!("all-MiniLM-L6-v2: {}", present(HAS_MINILM));
    println!("jina-clip-v1:  {}", present(HAS_JINA));
    println!("bioclip:       {}", present(HAS_BIOCLIP));
    println!("nomic v1.5:    {}", present(HAS_NOMIC));
    println!("PP-OCRv6:      {}", present(HAS_OCR));
    for size in &PE_CORE_SIZES {
        if size.compiled_in {
            println!("{}: compiled in", size.name);
        } else {
            println!("{0}: missing — run scripts/pe/export_onnx.py {0}, then rebuild", size.name);
        }
    }
    println!("clip models:   {}", CLIP_MODEL_NAMES.join(", "));
}

fn classify(setup: &ClassifierSetup, top: usize, photos: &[PathBuf]) -> Result<()> {
    let model = load_clip(&setup.model)?;
    let labels = read_labels_csv(&std::fs::read_to_string(&setup.labels).context("could not read the labels file")?)?;
    let classifier =
        ZeroShotClassifier::new(model.text.as_ref(), &labels, setup.wording.into(), setup.prompts.into(), model.logit_scale)?;

    for photo in photos {
        let started = Instant::now();
        let encoded = std::fs::read(photo).with_context(|| format!("could not read {}", photo.display()))?;
        let ranked = classifier.rank(&model.vision.embed_image(&encoded)?);

        println!("{} ({} ms)", photo.display(), started.elapsed().as_millis());
        for prediction in ranked.iter().take(top) {
            println!("  {:<16} cosine {:.4}  p {:.3}", prediction.class, prediction.cosine, prediction.probability);
        }
    }
    Ok(())
}

/// The extensions of the formats the vision model reads.
const IMAGE_EXTENSIONS: [&str; 8] = ["jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp", "gif"];

fn compare_items(items: &[String]) -> Result<()> {
    let is_image: Vec<bool> = items.iter().map(|item| Path::new(item).is_file()).collect();
    // Embedded as text, a mistyped path would still score against everything else.
    for (item, &is_image) in items.iter().zip(&is_image) {
        if !is_image && has_image_extension(item) {
            anyhow::bail!("{item} is named like an image, but there is no such file");
        }
    }
    let text_embedder = is_image.contains(&false).then(load_nomic_text).transpose()?;
    let image_embedder = is_image.contains(&true).then(load_nomic_vision).transpose()?;

    let mut embeddings = Vec::new();
    for (number, (item, &is_image)) in items.iter().zip(&is_image).enumerate() {
        let embedding = if is_image {
            let encoded = std::fs::read(item).with_context(|| format!("could not read {item}"))?;
            let embedder = image_embedder.as_ref().expect("loaded for images");
            embedder.embed_image(&encoded).with_context(|| format!("could not embed {item}"))?
        } else {
            let embedder = text_embedder.as_ref().expect("loaded for text");
            embedder.embed(item, TextRole::Query)?
        };
        let kind = if is_image { "image" } else { "text" };
        println!("[{}] {kind:<5} {item}", number + 1);
        embeddings.push(embedding);
    }

    println!();
    for (i, a) in embeddings.iter().enumerate() {
        for (j, b) in embeddings.iter().enumerate().skip(i + 1) {
            println!("[{}] ~ [{}]  {:.4}", i + 1, j + 1, cosine_similarity(a, b));
        }
    }
    Ok(())
}

fn has_image_extension(item: &str) -> bool {
    let extension = Path::new(item).extension().and_then(|extension| extension.to_str());
    extension.is_some_and(|extension| IMAGE_EXTENSIONS.iter().any(|known| extension.eq_ignore_ascii_case(known)))
}

fn read_text(image: &Path) -> Result<()> {
    let engine = OcrEngine::load()?;
    let encoded = std::fs::read(image).with_context(|| format!("could not read {}", image.display()))?;
    let lines = engine.read(&encoded).with_context(|| format!("could not read text in {}", image.display()))?;

    // A detected box in which nothing was recognised has no text to print.
    let texts: Vec<&str> = lines.iter().map(|line| line.text.as_str()).filter(|text| !text.trim().is_empty()).collect();
    match texts.len() {
        0 => println!("{}: no text detected", image.display()),
        1 => println!("{}: 1 line of text", image.display()),
        count => println!("{}: {count} lines of text", image.display()),
    }
    for text in texts {
        println!("  {text}");
    }
    Ok(())
}
