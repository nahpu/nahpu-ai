use anyhow::Result;

use nahpu_embed_burn::{BACKEND_NAME, HAS_MODEL};

fn main() -> Result<()> {
    println!("backend:   {BACKEND_NAME}");
    println!("model:     {}", describe_model_availability());
    Ok(())
}

fn describe_model_availability() -> &'static str {
    if HAS_MODEL {
        "compiled in"
    } else {
        "none — run scripts/fetch-model.sh, then rebuild"
    }
}
