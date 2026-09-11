use nahpu_embed_burn::{baseline_tokenizer_path, baseline_weights_path, BurnEmbedder};
use nahpu_embed_core::{cosine_similarity, matches_known_good, TextEmbedder, TextRole};

const REFERENCE: &str = include_str!("data/candle-reference.tsv");

#[test]
fn matches_the_candle_reference_embeddings() {
    let Some(embedder) = load_baseline_embedder() else {
        eprintln!("skipped: no model compiled in — run scripts/fetch-model.sh, then rebuild");
        return;
    };

    for (sentence, expected) in reference_embeddings() {
        let actual = embedder
            .embed(&sentence, TextRole::Document)
            .expect("embedding a reference sentence");

        assert_eq!(actual.len(), expected.len(), "dimensions for {sentence:?}");
        assert!(
            matches_known_good(&actual, &expected),
            "{sentence:?} drifted from the reference: cosine {}",
            cosine_similarity(&actual, &expected)
        );
    }
}

fn load_baseline_embedder() -> Option<BurnEmbedder> {
    let weights = std::fs::read(baseline_weights_path()?).ok()?;
    let tokenizer = std::fs::read(baseline_tokenizer_path()?).ok()?;
    BurnEmbedder::from_bytes(&weights, &tokenizer).ok()
}

fn reference_embeddings() -> Vec<(String, Vec<f32>)> {
    REFERENCE
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let (sentence, vector) = line
                .split_once('\t')
                .expect("reference row is sentence\\tvector");
            let vector = vector
                .split(',')
                .map(|x| x.parse().expect("float"))
                .collect();
            (sentence.to_string(), vector)
        })
        .collect()
}

#[test]
fn reference_data_is_actually_loaded() {
    let rows = reference_embeddings();
    assert_eq!(rows.len(), 3, "expected three reference sentences");
    for (sentence, vector) in rows {
        assert_eq!(vector.len(), 384, "dimensions for {sentence:?}");
    }
}

#[test]
fn parity_check_rejects_a_wrong_embedding() {
    let (_, reference) = reference_embeddings().remove(0);
    let (_, different) = reference_embeddings().remove(2);
    assert!(!matches_known_good(&different, &reference));
}
