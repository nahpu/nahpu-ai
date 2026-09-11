use anyhow::Result;

use nahpu_embed_burn::{
    baseline_tokenizer_path, baseline_weights_path, BurnEmbedder, BACKEND_NAME,
};
use nahpu_embed_core::{cosine_similarity, TextEmbedder, TextRole};

fn main() -> Result<()> {
    println!("backend:   {BACKEND_NAME}");

    let (Some(weights), Some(tokenizer)) = (baseline_weights_path(), baseline_tokenizer_path())
    else {
        println!("model:     none — run scripts/fetch-model.sh, then rebuild");
        return Ok(());
    };
    println!("model:     compiled in");

    embed_sample_sentences(weights, tokenizer)
}

fn embed_sample_sentences(weights_path: &str, tokenizer_path: &str) -> Result<()> {
    let embedder = BurnEmbedder::from_bytes(
        &std::fs::read(weights_path)?,
        &std::fs::read(tokenizer_path)?,
    )?;

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
