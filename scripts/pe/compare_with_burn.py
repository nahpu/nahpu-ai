"""Checks a Rust eval run of a PE-Core size against its PyTorch run, prediction by prediction.

    python3 scripts/pe/compare_with_burn.py RUST_RESULTS PYTORCH_RESULTS

Both folders hold the predictions/ that `cli eval --sweep` and eval_pytorch.py
write. Every setting they share is compared. Fails if fewer than 97% of top-1
predictions agree in any of them.

The gate is looser than the 99% scripts/reference/eval_reference.py sets for
jina-clip-v1. The two engines match on the same pixels (tests/pe_parity.rs),
but the Rust and PIL JPEG decoders and resize filters do not give the same
pixels: for L the image vectors agree to cosine 0.9977 or better over the 700
NEON photos. Over the five sizes that flips up to 2.7% of predictions with the
taxon wording, where two classes often score almost the same, and up to 1.9%
with the others.
"""

import csv
import sys
from pathlib import Path

REQUIRED_AGREEMENT = 0.97


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare(rust_rows, pytorch_rows):
    if len(rust_rows) != len(pytorch_rows):
        sys.exit(f"the runs score different numbers of photos: {len(rust_rows)} and {len(pytorch_rows)}")
    cosine_columns = [name for name in rust_rows[0] if name.startswith("cos:")]
    agree, largest_gap = 0, 0.0
    for rust, pytorch in zip(rust_rows, pytorch_rows):
        if rust["path"] != pytorch["path"]:
            sys.exit(f"the runs list different photos: {rust['path']} and {pytorch['path']}")
        agree += rust["predicted"] == pytorch["predicted"]
        largest_gap = max(largest_gap, max(abs(float(rust[c]) - float(pytorch[c])) for c in cosine_columns))
    return agree / len(rust_rows), largest_gap


def main(rust, pytorch):
    rust, pytorch = Path(rust) / "predictions", Path(pytorch) / "predictions"
    shared = sorted(file.name for file in rust.glob("*.csv") if (pytorch / file.name).exists())
    if not shared:
        sys.exit(f"{rust} and {pytorch} share no setting")

    worst = 1.0
    for name in shared:
        agreement, gap = compare(read_rows(rust / name), read_rows(pytorch / name))
        worst = min(worst, agreement)
        print(f"{name.removesuffix('.csv')}: top-1 agreement {agreement:.4f}, largest cosine gap {gap:.5f}")
    print(f"{len(shared)} settings, lowest agreement {worst:.4f}")
    sys.exit(0 if worst >= REQUIRED_AGREEMENT else 1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
