"""Zero-shot ranking through onnxruntime, mirroring the Rust ZeroShotClassifier.

The templates and wording must match rust/crates/nahpu_embed_core/src/zero_shot.rs.
If they drift, the cross-check in eval_reference.py fails on cosines.
"""

import csv

import numpy as np

PROMPT_SETS = {
    "bare": ["{}"],
    "photo": ["a photo of a {}."],
    "specimen": ["a photo of a museum specimen of a {}."],
    "ensemble": [
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


def read_labels(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def worded(label, wording):
    if wording == "common":
        return label["common_name"]
    if wording == "taxon":
        return label["taxon_name"]
    return f"{label['common_name']} ({label['taxon_name']})"


def class_vectors(clip, labels, wording, prompts):
    templates = PROMPT_SETS[prompts]
    texts = [t.replace("{}", worded(label, wording)) for label in labels for t in templates]
    vectors = clip.embed_texts(texts).reshape(len(labels), len(templates), -1).mean(axis=1)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
