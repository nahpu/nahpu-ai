"""Checks that the files written by `cli embed` line up with the photos they claim to be.

Scores zero-shot from the saved vectors alone. The result must equal what
`cli eval` reported for the same setting, so a row in the wrong place shows up
as a different score.

    python3 -I scripts/analysis/check_embeddings.py EMBEDDINGS_DIR RESULTS_DIR

RESULTS_DIR is the `cli eval` output for the same model, e.g. eval/results/bioclip.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(embeddings, results):
    embeddings, results = Path(embeddings), Path(results)
    meta = json.loads((embeddings / "meta.json").read_text())
    images = np.load(embeddings / "images.npy")
    labels = np.load(embeddings / "labels.npy")
    photos = read_rows(embeddings / "images.csv")
    classes = [row["class"] for row in read_rows(embeddings / "labels.csv")]
    failures = []

    if images.shape != (len(photos), meta["dimensions"]):
        failures.append(f"images.npy is {images.shape}, expected ({len(photos)}, {meta['dimensions']})")
    if labels.shape != (len(classes), meta["dimensions"]):
        failures.append(f"labels.npy is {labels.shape}, expected ({len(classes)}, {meta['dimensions']})")
    lengths = np.linalg.norm(np.concatenate([images, labels]), axis=1)
    if np.abs(lengths - 1).max() > 1e-4:
        failures.append(f"vectors are not unit length: worst is {lengths[np.abs(lengths - 1).argmax()]:.6f}")

    predicted = [classes[i] for i in (images @ labels.T).argmax(axis=1)]
    truth = [row["class"] for row in photos]
    per_class = {c: np.mean([p == t for p, t in zip(predicted, truth) if t == c]) for c in classes}
    macro = float(np.mean(list(per_class.values())))

    summary = json.loads((results / "metrics.json").read_text())
    run = next(
        r
        for r in summary["runs"]
        if (r["framing"], r["prompts"], r["wording"]) == (meta["framing"], meta["prompts"], meta["wording"])
    )
    print(f"{meta['model']}: macro accuracy from saved vectors {macro:.4f}, cli eval {run['macro_accuracy']:.4f}")
    if abs(macro - run["macro_accuracy"]) > 1e-9:
        failures.append("macro accuracy differs from cli eval")

    predictions = results / "predictions" / f"{meta['framing']}-{meta['prompts']}-{meta['wording']}.csv"
    if predictions.exists():
        by_path = {row["path"]: row["predicted"] for row in read_rows(predictions)}
        differ = [row["path"] for row, p in zip(photos, predicted) if by_path[row["path"]] != p]
        print(f"  per-photo predictions differing from {predictions.name}: {len(differ)}")
        if differ:
            failures.append(f"{len(differ)} photos predicted differently, first {differ[0]}")
    else:
        print(f"  no {predictions.name}; skipped the per-photo comparison")

    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
