"""Clusters the image vectors saved by `cli embed` and compares the clusters with
the taxonomy in the manifest. Uses no text, so prompt wording plays no part.

    python3 -I scripts/analysis/cluster.py MANIFEST OUT EMBEDDINGS_DIR [EMBEDDINGS_DIR ...]

Writes OUT/metrics.json, OUT/report.md and UMAP plots. Every metric is computed
on the full vectors; UMAP is only for the pictures.
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

SEED = 42
KMEANS_SEEDS = range(10)
FEW_LABELS = (1, 5, 10)
FEW_LABEL_DRAWS = 20
MIN_PHOTOS_PER_GENUS = 5
# Categorical slots 1-7 from the dataviz reference palette, light mode.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
OTHER_GREY = "#a3a29b"
INK, MUTED_INK = "#1a1a19", "#5c5b55"


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load(embeddings, manifest_rows):
    meta = json.loads((Path(embeddings) / "meta.json").read_text())
    vectors = np.load(Path(embeddings) / "images.npy").astype(np.float64)
    by_path = {row["path"]: row for row in manifest_rows}
    photos = [by_path[row["path"]] for row in read_rows(Path(embeddings) / "images.csv")]
    return meta, vectors, photos


def column(photos, name):
    return np.array([photo[name] for photo in photos])


def scores(truth, clusters):
    return {"ari": adjusted_rand_score(truth, clusters), "nmi": normalized_mutual_info_score(truth, clusters)}


def neighbours(vectors, k):
    """Indices of the k most similar other photos, nearest first."""
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    return np.argsort(-similarity, axis=1)[:, :k]


def vote(labels, nearest):
    """Majority class among the neighbours; a tie goes to the nearest tied class."""
    counts = Counter(labels[nearest])
    best = max(counts.values())
    return next(labels[j] for j in nearest if counts[labels[j]] == best)


def macro_accuracy(truth, predicted):
    return float(np.mean([np.mean(predicted[truth == c] == c) for c in np.unique(truth)]))


def unit(rows):
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


def clusters_vs_classes(vectors, classes):
    k = len(np.unique(classes))
    kmeans = [KMeans(k, n_init=10, random_state=seed).fit_predict(vectors) for seed in KMEANS_SEEDS]
    kmeans_scores = [scores(classes, c) for c in kmeans]
    agglomerative = AgglomerativeClustering(k, metric="cosine", linkage="average").fit_predict(vectors)
    hdbscan = HDBSCAN(min_cluster_size=10, copy=True).fit_predict(vectors)
    clustered = hdbscan >= 0

    return {
        "kmeans": {
            "ari_mean": float(np.mean([s["ari"] for s in kmeans_scores])),
            "ari_std": float(np.std([s["ari"] for s in kmeans_scores])),
            "nmi_mean": float(np.mean([s["nmi"] for s in kmeans_scores])),
            "nmi_std": float(np.std([s["nmi"] for s in kmeans_scores])),
            "confusion_seed_0": confusion(classes, kmeans[0]),
        },
        "agglomerative": {**scores(classes, agglomerative), "cluster_sizes": sorted(Counter(agglomerative.tolist()).values(), reverse=True)},
        "hdbscan": {
            "clusters": int(hdbscan.max() + 1),
            "noise_share": float(1 - clustered.mean()),
            **{f"{name}_all": value for name, value in scores(classes, hdbscan).items()},
            **{f"{name}_clustered": value for name, value in scores(classes[clustered], hdbscan[clustered]).items()},
        },
    }, kmeans[0]


def confusion(classes, clusters):
    """Rows are clusters, biggest first; columns are classes in sorted order."""
    names = sorted(np.unique(classes))
    rows = [[int(np.sum((clusters == c) & (classes == name))) for name in names] for c in np.unique(clusters)]
    return {"classes": names, "rows": sorted(rows, key=sum, reverse=True)}


def neighbour_accuracy(vectors, classes):
    nearest = neighbours(vectors, 5)
    knn = {}
    for k in (1, 5):
        predicted = np.array([vote(classes, row[:k]) for row in nearest])
        knn[f"k{k}"] = {
            "macro_accuracy": macro_accuracy(classes, predicted),
            "shrew_called_mouse_or_vole": int(np.sum((classes == "shrew") & (predicted == "mouse_or_vole"))),
            "mouse_or_vole_called_shrew": int(np.sum((classes == "mouse_or_vole") & (predicted == "shrew"))),
        }

    rng = np.random.default_rng(SEED)
    names = np.unique(classes)
    few = {}
    for n in FEW_LABELS:
        accuracies = []
        for _ in range(FEW_LABEL_DRAWS):
            labelled = np.zeros(len(classes), dtype=bool)
            for name in names:
                labelled[rng.choice(np.flatnonzero(classes == name), n, replace=False)] = True
            centres = unit(np.stack([vectors[labelled & (classes == name)].mean(axis=0) for name in names]))
            predicted = names[(vectors[~labelled] @ centres.T).argmax(axis=1)]
            accuracies.append(macro_accuracy(classes[~labelled], predicted))
        few[str(n)] = {"mean": float(np.mean(accuracies)), "std": float(np.std(accuracies))}

    centres = unit(np.stack([vectors[classes == name].mean(axis=0) for name in names]))
    similarity = centres @ centres.T
    pairs = sorted(
        ((float(similarity[i, j]), names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))),
        reverse=True,
    )
    return {
        "silhouette": float(silhouette_score(vectors, classes, metric="cosine")),
        "knn": knn,
        "few_label_nearest_centre": few,
        "centre_similarity_most_similar_first": [{"pair": f"{a} / {b}", "cosine": c} for c, a, b in pairs],
    }


def below_class(vectors, photos):
    classes, genera = column(photos, "class"), column(photos, "genus")
    results = {}
    for name in sorted(np.unique(classes)):
        counts = Counter(genera[classes == name])
        kept = sorted(g for g, n in counts.items() if n >= MIN_PHOTOS_PER_GENUS)
        if len(kept) < 2:
            continue
        mask = (classes == name) & np.isin(genera, kept)
        subset, truth = vectors[mask], genera[mask]
        clusters = KMeans(len(kept), n_init=10, random_state=SEED).fit_predict(subset)
        rng = np.random.default_rng(SEED)
        shuffled = [normalized_mutual_info_score(rng.permutation(truth), clusters) for _ in range(20)]
        predicted = np.array([vote(truth, row) for row in neighbours(subset, 1)])
        results[name] = {
            "genera": {g: counts[g] for g in kept},
            "photos": int(mask.sum()),
            "photos_in_class": int((classes == name).sum()),
            "kmeans_nmi": normalized_mutual_info_score(truth, clusters),
            "kmeans_nmi_shuffled_genera": float(np.mean(shuffled)),
            "knn1_macro_accuracy": macro_accuracy(truth, predicted),
            "chance_macro_accuracy": 1 / len(kept),
        }
    return results


def photo_setup(vectors, photos, kmeans_clusters):
    """Does the nearest photo of the same class share the photographer or view more than chance?"""
    classes = column(photos, "class")
    results = {
        "nmi_class_vs_creator": normalized_mutual_info_score(classes, column(photos, "creator")),
        "nmi_clusters_vs_creator": normalized_mutual_info_score(column(photos, "creator"), kmeans_clusters),
        "within_class": {},
        "within_genus": {},
    }
    genera = column(photos, "genus")
    for field in ("creator", "view"):
        values = column(photos, field)
        for name in sorted(np.unique(classes)):
            check = same_as_nearest(vectors, values, (classes == name) & (values != ""))
            if check:
                results["within_class"][f"{name} / {field}"] = check
    # Photographers may have covered different genera, so repeat inside one genus.
    creators = column(photos, "creator")
    for genus in sorted(np.unique(genera)):
        check = same_as_nearest(vectors, creators, (genera == genus) & (creators != ""))
        if check:
            results["within_genus"][f"{genus} / creator"] = check
    return results


def same_as_nearest(vectors, values, mask):
    counts = np.array(list(Counter(values[mask]).values()))
    if len(counts) < 2 or mask.sum() < 20:
        return None
    nearest = neighbours(vectors[mask], 1)[:, 0]
    return {
        "photos": int(mask.sum()),
        "values": len(counts),
        "same_as_nearest": float(np.mean(values[mask] == values[mask][nearest])),
        "chance": float(np.sum(counts * (counts - 1)) / (mask.sum() * (mask.sum() - 1))),
    }


def misfits(vectors, photos):
    """Photos whose five nearest neighbours all belong to another class."""
    classes = column(photos, "class")
    nearest = neighbours(vectors, 5)
    rows = []
    for i, row in enumerate(nearest):
        if not np.any(classes[row] == classes[i]):
            rows.append({
                "path": photos[i]["path"],
                "class": classes[i],
                "genus": photos[i]["genus"],
                "neighbours": Counter(classes[row].tolist()).most_common(1)[0][0],
            })
    return rows


def project(vectors):
    return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED, n_jobs=1).fit_transform(vectors)


def style(ax, title):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d9d8d2")


def plot_classes(models, out):
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 6), squeeze=False)
    for ax, (meta, points, photos) in zip(axes[0], models):
        classes = column(photos, "class")
        for slot, name in enumerate(sorted(np.unique(classes))):
            mine = points[classes == name]
            ax.scatter(*mine.T, s=22, c=COLOURS[slot], marker=MARKERS[slot], edgecolors="white", linewidths=0.5, label=name)
            ax.annotate(name, np.median(mine, axis=0), color=INK, fontsize=9, ha="center", weight="bold",
                        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8})
        style(ax, f"{meta['model']}: image vectors only, coloured by class")
    axes[0][0].legend(loc="lower left", fontsize=8, frameon=False, labelcolor=MUTED_INK)
    fig.text(0.01, 0.01, "UMAP of the full vectors (cosine, 15 neighbours, seed 42). Axes have no units.", color=MUTED_INK, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "umap-classes.png", dpi=150)
    plt.close(fig)


def plot_genera(meta, vectors, photos, genus_results, out):
    names = list(genus_results)
    columns = 4
    rows = -(-len(names) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4.2 * rows), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    classes, genera = column(photos, "class"), column(photos, "genus")
    for ax, name in zip(axes.flat, names):
        mask = classes == name
        points = project(vectors[mask])
        top = [g for g, _ in Counter(genera[mask]).most_common(3)]
        others = ~np.isin(genera[mask], top)
        ax.scatter(*points[others].T, s=16, c=OTHER_GREY, marker="o", edgecolors="white", linewidths=0.5, label="other genera")
        for slot, genus in enumerate(top):
            mine = genera[mask] == genus
            ax.scatter(*points[mine].T, s=22, c=COLOURS[slot], marker=MARKERS[slot], edgecolors="white", linewidths=0.5,
                       label=f"{genus} ({mine.sum()})")
        style(ax, name)
        ax.legend(loc="best", fontsize=7, frameon=False, labelcolor=MUTED_INK, prop={"style": "italic", "size": 7})
    fig.suptitle(f"{meta['model']}: each class on its own, coloured by genus (top 3)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / f"umap-genera-{meta['model']}.png", dpi=150)
    plt.close(fig)


def percent(value):
    return f"{value * 100:.1f}%"


def report(results):
    models = list(results)
    lines = ["# Clustering the image vectors", ""]
    lines += [f"Models: {', '.join(models)}. 700 NEON photos, 7 classes, centre-crop. No text is used.", ""]
    header = "| | " + " | ".join(models) + " |"
    rule = "|---|" + "---|" * len(models)

    def row(label, value):
        return f"| {label} | " + " | ".join(value(results[m]) for m in models) + " |"

    lines += ["## Clusters vs classes", "", "ARI and NMI: 0 is random, 1 is a perfect match.", "", header, rule]
    lines.append(row("k-means k=7, ARI (10 seeds)", lambda r: f"{r['clusters']['kmeans']['ari_mean']:.3f} ± {r['clusters']['kmeans']['ari_std']:.3f}"))
    lines.append(row("k-means k=7, NMI", lambda r: f"{r['clusters']['kmeans']['nmi_mean']:.3f} ± {r['clusters']['kmeans']['nmi_std']:.3f}"))
    lines.append(row("Agglomerative, ARI / NMI", lambda r: f"{r['clusters']['agglomerative']['ari']:.3f} / {r['clusters']['agglomerative']['nmi']:.3f}"))
    lines.append(row("Agglomerative, cluster sizes", lambda r: ", ".join(map(str, r["clusters"]["agglomerative"]["cluster_sizes"]))))
    lines.append(row("HDBSCAN, clusters found", lambda r: str(r["clusters"]["hdbscan"]["clusters"])))
    lines.append(row("HDBSCAN, left as noise", lambda r: percent(r["clusters"]["hdbscan"]["noise_share"])))
    lines.append(row("HDBSCAN, ARI / NMI (clustered photos)", lambda r: f"{r['clusters']['hdbscan']['ari_clustered']:.3f} / {r['clusters']['hdbscan']['nmi_clustered']:.3f}"))

    for m in models:
        c = results[m]["clusters"]["kmeans"]["confusion_seed_0"]
        lines += ["", f"k-means (seed 0) for {m}: rows are clusters, columns are classes.", ""]
        lines += ["| cluster | " + " | ".join(c["classes"]) + " |", "|---|" + "---|" * len(c["classes"])]
        lines += [f"| {i + 1} | " + " | ".join(map(str, r)) + " |" for i, r in enumerate(c["rows"])]

    lines += ["", "## Neighbours", "", header, rule]
    lines.append(row("Silhouette by class (−1 to 1)", lambda r: f"{r['neighbours']['silhouette']:.3f}"))
    for k in ("k1", "k5"):
        lines.append(row(f"Leave-one-out {k[1]}-NN, macro accuracy", lambda r, k=k: percent(r["neighbours"]["knn"][k]["macro_accuracy"])))
    for n in FEW_LABELS:
        lines.append(row(f"{n} labelled per class, nearest centre", lambda r, n=n: f"{percent(r['neighbours']['few_label_nearest_centre'][str(n)]['mean'])} ± {r['neighbours']['few_label_nearest_centre'][str(n)]['std'] * 100:.1f}"))
    lines.append(row("Zero-shot, best setting (from 08)", lambda r: r["zero_shot"]))
    lines.append(row("1-NN: shrews called mouse or vole", lambda r: str(r["neighbours"]["knn"]["k1"]["shrew_called_mouse_or_vole"])))
    lines.append(row("1-NN: mice or voles called shrew", lambda r: str(r["neighbours"]["knn"]["k1"]["mouse_or_vole_called_shrew"])))
    lines.append(row("Most similar class centres", lambda r: r["neighbours"]["centre_similarity_most_similar_first"][0]["pair"]))
    lines.append(row("Shrew / mouse-or-vole rank of 21 pairs", lambda r: str(1 + [p["pair"] for p in r["neighbours"]["centre_similarity_most_similar_first"]].index("mouse_or_vole / shrew"))))

    lines += ["", "## Below class level", "", f"Only genera with {MIN_PHOTOS_PER_GENUS}+ photos. NMI baseline: genus labels shuffled 20 times.", ""]
    lines += ["| Class | Genera kept | Photos kept | Model | k-means NMI (shuffled) | 1-NN genus accuracy (chance) |", "|---|---|---|---|---|---|"]
    for name in results[models[0]]["below_class"]:
        for m in models:
            g = results[m]["below_class"][name]
            lines.append(
                f"| {name} | {', '.join(f'*{k}* {v}' for k, v in g['genera'].items())} | {g['photos']}/{g['photos_in_class']} | {m} | "
                f"{g['kmeans_nmi']:.3f} ({g['kmeans_nmi_shuffled_genera']:.3f}) | {percent(g['knn1_macro_accuracy'])} ({percent(g['chance_macro_accuracy'])}) |"
            )

    lines += ["", "## Photo setup", "", "Within one class, how often the nearest photo shares the photographer or view, against chance.", ""]
    lines += ["| Group / field | Photos | Values | Model | Same as nearest | Chance |", "|---|---|---|---|---|---|"]
    for scope in ("within_class", "within_genus"):
        for key in results[models[0]]["photo_setup"][scope]:
            for m in models:
                s = results[m]["photo_setup"][scope][key]
                group = key if scope == "within_class" else f"*{key.split(' / ')[0]}* / creator"
                lines.append(f"| {group} | {s['photos']} | {s['values']} | {m} | {percent(s['same_as_nearest'])} | {percent(s['chance'])} |")
    lines += ["", "Rows in italics are inside one genus, which removes genus as the reason."]
    lines += ["", header, rule]
    lines.append(row("NMI, class vs photographer", lambda r: f"{r['photo_setup']['nmi_class_vs_creator']:.3f}"))
    lines.append(row("NMI, k-means clusters vs photographer", lambda r: f"{r['photo_setup']['nmi_clusters_vs_creator']:.3f}"))
    lines += ["", "Image size is 512×341 for 699 of 700 photos, so it is not tested."]

    lines += ["", "## Misfits", "", "Photos whose 5 nearest neighbours all belong to another class.", ""]
    for m in models:
        lines += [f"**{m}:** {len(results[m]['misfits'])} photos.", ""]
        if results[m]["misfits"]:
            lines += ["| Photo | Class | Genus | Neighbours are |", "|---|---|---|---|"]
            lines += [f"| `{f['path']}` | {f['class']} | *{f['genus']}* | {f['neighbours']} |" for f in results[m]["misfits"]]
            lines.append("")

    lines += ["## Plots", "", "![UMAP by class](umap-classes.png)", ""]
    lines += [f"![{m} genera](umap-genera-{m}.png)" for m in models]
    return "\n".join(lines) + "\n"


def zero_shot_from_eval(meta):
    summary = json.loads(Path("eval/results", meta["model"], "metrics.json").read_text())
    return percent(max(r["macro_accuracy"] for r in summary["runs"] if r["framing"] == meta["framing"]))


def main(manifest, out, *embeddings):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_rows(manifest)
    results, projected = {}, []
    for directory in embeddings:
        meta, vectors, photos = load(directory, manifest_rows)
        classes = column(photos, "class")
        print(f"{meta['model']}: {vectors.shape}", file=sys.stderr)
        cluster_results, kmeans_clusters = clusters_vs_classes(vectors, classes)
        genus_results = below_class(vectors, photos)
        results[meta["model"]] = {
            "meta": meta,
            "clusters": cluster_results,
            "neighbours": neighbour_accuracy(vectors, classes),
            "below_class": genus_results,
            "photo_setup": photo_setup(vectors, photos, kmeans_clusters),
            "misfits": misfits(vectors, photos),
            "zero_shot": zero_shot_from_eval(meta),
        }
        projected.append((meta, project(vectors), photos))
        plot_genera(meta, vectors, photos, genus_results, out)
    plot_classes(projected, out)

    (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "report.md").write_text(report(results))
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
