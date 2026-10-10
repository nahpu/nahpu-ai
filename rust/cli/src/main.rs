use anyhow::Result;

use nahpu_embed_burn::{load_minilm, BACKEND_NAME, HAS_MINILM};
use nahpu_embed_core::{cosine_similarity, TextRole};

fn main() -> Result<()> {
    println!("backend:   {BACKEND_NAME}");

    if !HAS_MINILM {
        println!("model:     none — run scripts/fetch-model.sh, then rebuild");
        return Ok(());
    }
    println!("model:     compiled in");

    embed_sample_sentences()
}

fn embed_sample_sentences() -> Result<()> {
    let embedder = load_minilm()?;

    let bird = embedder.embed(
        "A small brown bird perched on a branch.",
        TextRole::Document,
    )?;
    let sparrow = embedder.embed("A tiny sparrow sitting in a tree.", TextRole::Document)?;
    let engine = embedder.embed("Diesel engine maintenance schedule.", TextRole::Document)?;

    println!();
    println!("dimensions:      {}", bird.len());
    println!("unit length:     {:.6}", magnitude(&bird));
    println!("bird ~ sparrow:  {:.4}", cosine_similarity(&bird, &sparrow));
    println!("bird ~ engine:   {:.4}", cosine_similarity(&bird, &engine));
    Ok(())
}

fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
