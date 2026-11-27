"""Scores one PE-Core size on a labelled manifest in PyTorch, with no Burn involved.

    scripts/pe/.venv/bin/python -I scripts/pe/eval_pytorch.py MODEL MANIFEST LABELS RESULTS_OUT EMBEDDINGS_OUT

MODEL is one of the names in pe_core.py, from pe-core-t16-384 to pe-core-g14-448.

This is what a Burn run of the same size is checked against
(compare_with_burn.py). It writes the files the Rust commands write, so the
analysis scripts read them unchanged:

    RESULTS_OUT/     metrics.json, report.md and predictions/, as `cli eval --sweep`
    EMBEDDINGS_OUT/  images.npy, labels.npy, images.csv, labels.csv and meta.json,
                     as `cli embed`, for centre-crop and its best prompt setting

Uses a CUDA GPU when there is one. G runs there in half precision; one photo and
three prompts are also run on the CPU in single precision, and the run stops if
the two differ.
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from pe_core import CHECKPOINTS, FRAMINGS, PeCore, normalise_to_chw, upright_rgb
from zero_shot import PROMPT_SETS, class_vectors, read_labels

WORDINGS = ["common", "taxon", "both"]
SAVED_FRAMING = "centre-crop"
DEFAULT_SETTING = ("centre-crop", "ensemble", "common")
PRECISION_PROMPTS = ["a photo of a mouse.", "a photo of a museum specimen of a shrew.", "salamander"]
REQUIRED_PRECISION_COSINE = 0.999


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_squash_is_the_models_own(clip, photo):
    ours = normalise_to_chw(FRAMINGS["squash"](upright_rgb(photo), clip.size))
    gap = float(np.abs(clip.own_transform(upright_rgb(photo)).numpy() - ours).max())
    if gap > 1e-5:
        sys.exit(f"squash differs from open_clip's own transform by {gap}")


def move_to_gpu(clip, photo):
    """Returns the lowest cosine between half and single precision, or None if the GPU run is not in half."""
    def probe():
        return np.concatenate([clip.embed_image(photo)[None], clip.embed_texts(PRECISION_PROMPTS)])

    exact = probe()
    clip.to_gpu()
    if not clip.half:
        return None
    cosine = float((exact * probe()).sum(axis=1).min())
    if not cosine >= REQUIRED_PRECISION_COSINE:
        sys.exit(f"half precision drifted from single precision: cosine {cosine}")
    return cosine


def embed_all(clip, base, photos, framing):
    vectors, milliseconds = [], []
    for i, photo in enumerate(photos):
        started = time.perf_counter()
        vectors.append(clip.embed_image(base / photo["path"], framing))
        milliseconds.append((time.perf_counter() - started) * 1000)
        if (i + 1) % 100 == 0:
            print(f"  {framing} {i + 1}/{len(photos)}", file=sys.stderr)
    # Single precision, so the scores are those of the saved files.
    return np.stack(vectors).astype(np.float32), milliseconds


def score(photos, vectors, classes, class_names, setting, out):
    framing, prompts, wording = setting
    cosines = vectors @ classes.T
    ranked = np.argsort(-cosines, axis=1, kind="stable")
    truth = np.array([class_names.index(photo["class"]) for photo in photos])
    confusion = np.zeros((len(class_names), len(class_names)), int)
    np.add.at(confusion, (truth, ranked[:, 0]), 1)

    with open(out / "predictions" / f"{framing}-{prompts}-{wording}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "truth", "predicted", "correct"] + [f"cos:{name}" for name in class_names])
        for photo, t, order, row in zip(photos, truth, ranked, cosines):
            predicted = class_names[order[0]]
            correct = str(predicted == class_names[t]).lower()
            writer.writerow([photo["path"], class_names[t], predicted, correct] + [f"{c:.6f}" for c in row])

    per_class = {name: confusion[i, i] / confusion[i].sum() for i, name in enumerate(class_names) if confusion[i].sum()}
    return {
        "framing": framing,
        "prompts": prompts,
        "wording": wording,
        "top1": float(np.mean(ranked[:, 0] == truth)),
        "top3": float(np.mean((ranked[:, :3] == truth[:, None]).any(axis=1))),
        "macro_accuracy": float(np.mean(list(per_class.values()))),
        "per_class_accuracy": {name: float(per_class[name]) for name in sorted(per_class)},
        "confusion": confusion.tolist(),
    }


def percentile(values, fraction):
    return sorted(values)[round((len(values) - 1) * fraction)]


def percent(fraction):
    return f"{fraction * 100:.1f}%"


def report(summary):
    """The layout of `cli eval`'s report, so the two read side by side."""
    md = [f"# Zero-shot evaluation: {summary['model']}", ""]
    md += [f"- Backend: {summary['backend']}, by `scripts/pe/eval_pytorch.py`, not by `cli eval`"]
    md += [f"- Checkpoint: `{summary['checkpoint']}`"]
    md += [f"- Manifest: `{summary['manifest']}`, {summary['images']} images"]
    md += [
        f"- Model load {summary['load_seconds']:.2f} s; per image (decode + preprocess + embed) "
        f"median {summary['image_ms_median']:.0f} ms, p95 {summary['image_ms_p95']:.0f} ms"
    ]
    if summary["half_precision_cosine"] is not None:
        md += [
            "- Half precision. Against single precision on the CPU, for one photo and three prompts: "
            f"cosine {summary['half_precision_cosine']:.6f} or better"
        ]
    md += ["- `squash` is the model's own framing. `centre-crop` and `pad` are the ones `cli eval` gives every model."]

    md += ["", "## All settings, best macro accuracy first", ""]
    md += ["| Framing | Prompts | Wording | Top-1 | Top-3 | Macro |", "|---|---|---|---|---|---|"]
    ranked = sorted(summary["runs"], key=lambda run: -run["macro_accuracy"])
    for run in ranked:
        cells = [run["framing"], run["prompts"], run["wording"], percent(run["top1"]), percent(run["top3"]), percent(run["macro_accuracy"])]
        md += ["| " + " | ".join(cells) + " |"]

    default = next(run for run in summary["runs"] if (run["framing"], run["prompts"], run["wording"]) == DEFAULT_SETTING)
    md += detail(summary, default, "Default setting (centre-crop, ensemble, common)")
    if ranked[0] is not default:
        md += detail(summary, ranked[0], "Best setting")
    return "\n".join(md) + "\n"


def detail(summary, run, title):
    classes = summary["classes"]
    md = ["", f"## {title}: {run['framing']} / {run['prompts']} / {run['wording']}", ""]
    md += ["| Class | Images | Accuracy |", "|---|---|---|"]
    md += [f"| {c} | {summary['images_per_class'][c]} | {percent(run['per_class_accuracy'][c])} |" for c in classes if c in run["per_class_accuracy"]]
    md += ["", "Confusion matrix (rows = truth, columns = predicted):", ""]
    md += ["| | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    md += [f"| **{c}** | " + " | ".join(map(str, row)) + " |" for c, row in zip(classes, run["confusion"])]
    return md


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def git_commit():
    """As `cli embed`: "unknown" when git cannot say."""
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    except OSError:
        return "unknown"
    return git.stdout.strip() if git.returncode == 0 else "unknown"


def save_embeddings(out, summary, photos, class_names, images, labels, setting):
    """The files `cli embed` writes, so check_embeddings.py and cluster.py read them."""
    np.save(out / "images.npy", images)
    np.save(out / "labels.npy", labels)
    with open(out / "images.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["row", "path", "class"]] + [[i, p["path"], p["class"]] for i, p in enumerate(photos)])
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["row", "class"]] + [[i, name] for i, name in enumerate(class_names)])
    write_json(out / "meta.json", {
        "model": summary["model"],
        "backend": summary["backend"],
        "manifest": summary["manifest"],
        "framing": setting["framing"],
        "prompts": setting["prompts"],
        "wording": setting["wording"],
        "images": len(photos),
        "dimensions": int(images.shape[1]),
        "git_commit": git_commit(),
    })


def main(name, manifest, labels_path, results, embeddings):
    if name not in CHECKPOINTS:
        sys.exit(f"unknown model {name!r}; known: {', '.join(CHECKPOINTS)}")
    results, embeddings = Path(results), Path(embeddings)
    (results / "predictions").mkdir(parents=True, exist_ok=True)
    embeddings.mkdir(parents=True, exist_ok=True)
    base = Path(manifest).parent
    photos = read_rows(manifest)
    labels = read_labels(labels_path)
    class_names = [label["class"] for label in labels]

    started = time.perf_counter()
    clip = PeCore(name)
    check_squash_is_the_models_own(clip, base / photos[0]["path"])
    half_precision_cosine = move_to_gpu(clip, base / photos[0]["path"]) if torch.cuda.is_available() else None
    load_seconds = time.perf_counter() - started

    classes = {
        (prompts, wording): class_vectors(clip, labels, wording, prompts).astype(np.float32)
        for prompts in PROMPT_SETS
        for wording in WORDINGS
    }
    runs, timings, saved = [], [], None
    for framing in FRAMINGS:
        vectors, milliseconds = embed_all(clip, base, photos, framing)
        timings = timings or milliseconds
        runs += [score(photos, vectors, classes[key], class_names, (framing, *key), results) for key in classes]
        if framing == SAVED_FRAMING:
            saved = vectors

    summary = {
        "model": name,
        "backend": f"pytorch-{clip.device}" + ("-half-precision" if clip.half else ""),
        "checkpoint": "@".join(CHECKPOINTS[name]),
        "manifest": Path(manifest).as_posix(),
        "images": len(photos),
        "images_per_class": {c: n for c in sorted(class_names) if (n := sum(p["class"] == c for p in photos))},
        "load_seconds": load_seconds,
        "image_ms_median": percentile(timings, 0.5),
        "image_ms_p95": percentile(timings, 0.95),
        "half_precision_cosine": half_precision_cosine,
        "classes": class_names,
        "runs": runs,
    }
    write_json(results / "metrics.json", summary)
    (results / "report.md").write_text(report(summary), encoding="utf-8", newline="\n")

    best = max((run for run in runs if run["framing"] == SAVED_FRAMING), key=lambda run: run["macro_accuracy"])
    save_embeddings(embeddings, summary, photos, class_names, saved, classes[best["prompts"], best["wording"]], best)
    print(f"wrote {results} and {embeddings}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(*sys.argv[1:])
