"""Checks a Rust eval run against onnxruntime, prediction by prediction.

    scripts/reference/.venv/bin/python -I scripts/reference/eval_reference.py \\
        MANIFEST LABELS PREDICTIONS_CSV PROMPTS WORDING

Centre-crop only, since that is what the reference transform does. Fails if
fewer than 99% of top-1 predictions agree.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jina_clip import JinaClip
from zero_shot import class_vectors, read_labels

REQUIRED_AGREEMENT = 0.99


def main(manifest, labels_path, predictions, prompts, wording):
    clip = JinaClip()
    labels = read_labels(labels_path)
    classes = class_vectors(clip, labels, wording, prompts)
    names = [label["class"] for label in labels]
    base = Path(manifest).parent

    with open(predictions, newline="") as f:
        rust_rows = list(csv.DictReader(f))

    agree, cosine_gaps, disagreements = 0, [], []
    for i, row in enumerate(rust_rows):
        cosines = classes @ clip.embed_image(base / row["path"])
        ours = names[int(np.argmax(cosines))]
        rust = np.array([float(row[f"cos:{name}"]) for name in names])
        cosine_gaps.append(float(np.abs(rust - cosines).max()))
        if ours == row["predicted"]:
            agree += 1
        else:
            disagreements.append((row["path"], row["predicted"], ours, float(np.sort(cosines)[-1] - np.sort(cosines)[-2])))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rust_rows)}", file=sys.stderr)

    agreement = agree / len(rust_rows)
    print(f"top-1 agreement {agree}/{len(rust_rows)} = {agreement:.4f}")
    print(f"cosine gap: max {max(cosine_gaps):.5f}, mean {np.mean(cosine_gaps):.5f}")
    for path, rust, ours, margin in disagreements:
        print(f"  differs: {path}: rust {rust}, onnxruntime {ours} (onnxruntime margin {margin:.5f})")
    sys.exit(0 if agreement >= REQUIRED_AGREEMENT else 1)


if __name__ == "__main__":
    main(*sys.argv[1:])
