"""Picks the best eval setting on one half of the photos and scores it on the other.

Choosing the best of many settings on the same photos you report on flatters
the result. This splits by the manifest's draw_rank (even vs odd), so each half
keeps the species spread.

    python3 scripts/reference/held_out.py MANIFEST RESULTS_DIR
"""

import csv
import sys
from pathlib import Path


def macro_accuracy(rows):
    by_class = {}
    for row in rows:
        by_class.setdefault(row["truth"], []).append(row["correct"] == "true")
    return sum(sum(v) / len(v) for v in by_class.values()) / len(by_class)


def main(manifest, results):
    with open(manifest, newline="") as f:
        rank = {row["path"]: int(row["draw_rank"]) for row in csv.DictReader(f)}

    scores = []
    for file in sorted(Path(results, "predictions").glob("*.csv")):
        with open(file, newline="") as f:
            rows = list(csv.DictReader(f))
        tune = [r for r in rows if rank[r["path"]] % 2 == 0]
        test = [r for r in rows if rank[r["path"]] % 2 == 1]
        scores.append((macro_accuracy(tune), macro_accuracy(test), macro_accuracy(rows), file.stem))

    tune, test, full, name = max(scores)
    print(f"picked on even half: {name} (even {tune:.3f})")
    print(f"  scored on odd half: {test:.3f}   (all 700: {full:.3f})")
    tune, test, full, name = max((b, a, c, n) for a, b, c, n in scores)
    print(f"picked on odd half:  {name} (odd {tune:.3f})")
    print(f"  scored on even half: {test:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
