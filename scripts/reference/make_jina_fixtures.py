"""Writes the reference data the Rust jina parity tests read.

Run from the repo root after scripts/fetch-model.sh:
    scripts/reference/.venv/bin/python -I scripts/reference/make_jina_fixtures.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jina_clip import MODELS, ROOT, JinaClip, resize_and_centre_crop, upright_rgb

OUT = ROOT / "rust/crates/nahpu_embed_burn/tests/data"
FIXTURES = ["fixture-neon-shrew", "fixture-neon-beetle"]
PROMPTS = [
    "a photo of a mouse.",
    "a photo of a museum specimen of a shrew.",
    "salamander",
]


def vector_text(vector):
    return ",".join(f"{x:.8e}" for x in vector)


def main():
    clip = JinaClip()

    rows = ["# prompt\ttoken ids\tunit vector (onnxruntime, jina-clip-v1 text)"]
    for prompt, vector in zip(PROMPTS, clip.embed_texts(PROMPTS)):
        ids = clip.token_ids([prompt])[0]
        rows.append(f"{prompt}\t{','.join(map(str, ids))}\t{vector_text(vector)}")
    (OUT / "jina-text-reference.tsv").write_text("\n".join(rows) + "\n")

    rows = ["# fixture\tunit vector (onnxruntime on PIL-preprocessed pixels)"]
    for stem in FIXTURES:
        square = resize_and_centre_crop(upright_rgb(MODELS / f"{stem}.jpg"))
        (OUT / f"{stem}.pixels.rgb").write_bytes(np.asarray(square, np.uint8).tobytes())
        rows.append(f"{stem}\t{vector_text(clip.embed_image(MODELS / f'{stem}.jpg'))}")
    (OUT / "jina-image-reference.tsv").write_text("\n".join(rows) + "\n")

    print(f"wrote reference data to {OUT}")


if __name__ == "__main__":
    main()
