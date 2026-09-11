//! Scratch entry point for trying engines by hand.

use anyhow::Result;

fn main() -> Result<()> {
    println!("backend:   {}", nahpu_embed_burn::BACKEND_NAME);
    println!(
        "model:     {}",
        if nahpu_embed_burn::HAS_MODEL {
            "compiled in"
        } else {
            "none — run scripts/fetch-model.sh, then rebuild"
        }
    );
    Ok(())
}
