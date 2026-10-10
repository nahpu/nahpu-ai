mod eval;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

use nahpu_embed_burn::{load_clip, BACKEND_NAME, CLIP_MODEL_NAMES, HAS_JINA, HAS_MINILM};
use nahpu_embed_core::preprocess::Framing;
use nahpu_embed_core::zero_shot::{read_labels_csv, LabelWording, PromptSet, ZeroShotClassifier};

#[derive(Parser)]
#[command(about = "Try NAHPU embedding models by hand")]
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
    }
}

fn report_what_is_available() {
    let present = |yes: bool| if yes { "compiled in" } else { "missing — run scripts/fetch-model.sh, then rebuild" };
    println!("backend:       {BACKEND_NAME}");
    println!("all-MiniLM-L6-v2: {}", present(HAS_MINILM));
    println!("jina-clip-v1:  {}", present(HAS_JINA));
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
