//! Zero-shot classification: embed a prompt per class, then rank classes by
//! cosine against an image vector. No training, so labels can change freely.

use anyhow::{bail, Context, Result};

use crate::{cosine_similarity, normalise_to_unit_length, Embedding, TextEmbedder, TextRole};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Label {
    pub class: String,
    pub common_name: String,
    pub taxon_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelWording {
    Common,
    Taxon,
    Both,
}

impl Label {
    pub fn worded(&self, wording: LabelWording) -> String {
        match wording {
            LabelWording::Common => self.common_name.clone(),
            LabelWording::Taxon => self.taxon_name.clone(),
            LabelWording::Both => format!("{} ({})", self.common_name, self.taxon_name),
        }
    }
}

/// Reads `class,common_name,taxon_name` rows. Plain commas only, no quoting.
pub fn read_labels_csv(text: &str) -> Result<Vec<Label>> {
    let mut rows = text.lines().filter(|line| !line.trim().is_empty());
    let header = rows.next().context("labels file is empty")?;
    if header.trim() != "class,common_name,taxon_name" {
        bail!("expected header class,common_name,taxon_name, got {header:?}");
    }

    rows.map(|row| match row.split(',').map(str::trim).collect::<Vec<_>>()[..] {
        [class, common_name, taxon_name] => Ok(Label {
            class: class.to_string(),
            common_name: common_name.to_string(),
            taxon_name: taxon_name.to_string(),
        }),
        _ => bail!("expected three columns in {row:?}"),
    })
    .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptSet {
    Bare,
    Photo,
    Specimen,
    Ensemble,
}

impl PromptSet {
    pub const ALL: [PromptSet; 4] = [Self::Bare, Self::Photo, Self::Specimen, Self::Ensemble];

    pub fn templates(self) -> &'static [&'static str] {
        match self {
            Self::Bare => &["{}"],
            Self::Photo => &["a photo of a {}."],
            Self::Specimen => &["a photo of a museum specimen of a {}."],
            Self::Ensemble => &[
                "a photo of a {}.",
                "a photo of a museum specimen of a {}.",
                "a photo of a preserved {}.",
                "a natural history collection specimen of a {}.",
                "a close-up photo of a {}.",
                "a specimen photo of a {} next to a scale bar.",
                "a photo of a dead {}.",
                "a lab photo of a {}.",
            ],
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bare => "bare",
            Self::Photo => "photo",
            Self::Specimen => "specimen",
            Self::Ensemble => "ensemble",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    pub class: String,
    pub cosine: f32,
    pub probability: f32,
}

pub struct ZeroShotClassifier {
    classes: Vec<String>,
    class_vectors: Vec<Embedding>,
    logit_scale: f32,
}

impl ZeroShotClassifier {
    /// Each class vector is the mean of its unit-length prompt vectors,
    /// rescaled to unit length, as in the CLIP paper's prompt ensembling.
    pub fn new(
        text: &dyn TextEmbedder,
        labels: &[Label],
        wording: LabelWording,
        prompts: PromptSet,
        logit_scale: f32,
    ) -> Result<Self> {
        let templates = prompts.templates();
        let prompt_texts: Vec<String> = labels
            .iter()
            .flat_map(|label| {
                let name = label.worded(wording);
                templates.iter().map(move |template| template.replace("{}", &name))
            })
            .collect();

        let prompt_vectors = text.embed_batch(&prompt_texts, TextRole::Query)?;
        let class_vectors = prompt_vectors.chunks(templates.len()).map(mean_direction).collect();

        Ok(Self {
            classes: labels.iter().map(|label| label.class.clone()).collect(),
            class_vectors,
            logit_scale,
        })
    }

    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// One unit-length vector per class, in `classes` order.
    pub fn class_vectors(&self) -> &[Embedding] {
        &self.class_vectors
    }

    /// Best class first. Probabilities are a softmax over `logit_scale * cosine`.
    pub fn rank(&self, image: &[f32]) -> Vec<Prediction> {
        let cosines: Vec<f32> = self.class_vectors.iter().map(|class| cosine_similarity(image, class)).collect();
        let probabilities = softmax(&cosines, self.logit_scale);

        let mut ranked: Vec<Prediction> = self
            .classes
            .iter()
            .zip(cosines)
            .zip(probabilities)
            .map(|((class, cosine), probability)| Prediction {
                class: class.clone(),
                cosine,
                probability,
            })
            .collect();
        ranked.sort_by(|a, b| b.cosine.total_cmp(&a.cosine));
        ranked
    }
}

fn mean_direction(vectors: &[Embedding]) -> Embedding {
    let width = vectors[0].len();
    let mut total = vec![0.0; width];
    for vector in vectors {
        for (sum, value) in total.iter_mut().zip(normalise_to_unit_length(vector.clone())) {
            *sum += value;
        }
    }
    normalise_to_unit_length(total)
}

fn softmax(cosines: &[f32], scale: f32) -> Vec<f32> {
    let largest = cosines.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f32> = cosines.iter().map(|c| ((c - largest) * scale).exp()).collect();
    let total: f32 = exponentials.iter().sum();
    exponentials.into_iter().map(|e| e / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gives each known word its own axis, so cosines are easy to predict.
    struct WordAxes;

    impl TextEmbedder for WordAxes {
        fn name(&self) -> &str {
            "word-axes"
        }

        fn dimensions(&self) -> usize {
            3
        }

        fn embed(&self, text: &str, _role: TextRole) -> Result<Embedding> {
            let axis = |word: &str| if text.contains(word) { 1.0 } else { 0.0 };
            Ok(vec![axis("shrew"), axis("frog"), axis("beetle")])
        }
    }

    fn labels() -> Vec<Label> {
        read_labels_csv("class,common_name,taxon_name\nshrew,shrew,Soricidae\nfrog,frog,Anura\nbeetle,beetle,Carabidae\n")
            .unwrap()
    }

    #[test]
    fn reads_the_labels_file() {
        let labels = labels();
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[1].taxon_name, "Anura");
    }

    #[test]
    fn rejects_a_labels_file_without_the_header() {
        assert!(read_labels_csv("shrew,shrew,Soricidae\n").is_err());
    }

    #[test]
    fn wording_puts_both_names_in_the_prompt() {
        assert_eq!(labels()[0].worded(LabelWording::Both), "shrew (Soricidae)");
    }

    #[test]
    fn ranks_the_closest_class_first_with_probabilities_summing_to_one() {
        let classifier =
            ZeroShotClassifier::new(&WordAxes, &labels(), LabelWording::Common, PromptSet::Photo, 100.0).unwrap();

        let ranked = classifier.rank(&[0.1, 0.9, 0.2]);

        assert_eq!(ranked[0].class, "frog");
        assert_eq!(ranked[2].class, "shrew");
        let total: f32 = ranked.iter().map(|p| p.probability).sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ensemble_averages_prompt_directions() {
        let averaged = mean_direction(&[vec![2.0, 0.0], vec![0.0, 1.0]]);
        let half = std::f32::consts::FRAC_1_SQRT_2;
        assert!((averaged[0] - half).abs() < 1e-6 && (averaged[1] - half).abs() < 1e-6);
    }

    #[test]
    fn every_template_has_one_placeholder() {
        for prompts in PromptSet::ALL {
            assert!(prompts.templates().iter().all(|t| t.matches("{}").count() == 1));
        }
    }
}
