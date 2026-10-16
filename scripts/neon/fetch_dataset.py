"""Build the NEON specimen-photo test set for the zero-shot eval.

The manifest is the lock file. If data/neon/manifest.csv exists, the script
fetches exactly the images it lists. Use --resample to draw a new sample from
the Darwin Core Archives.

    python -I scripts/neon/fetch_dataset.py             # sample if needed, fetch
    python -I scripts/neon/fetch_dataset.py --dry-run   # show candidate counts
    python -I scripts/neon/fetch_dataset.py --smoke     # 5 per class subset
    python -I scripts/neon/fetch_dataset.py --verify    # re-hash saved images
"""

import argparse
import csv
import datetime as dt
import hashlib
import io
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import requests
import yaml
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "neon"
CACHE = DATA / "cache"
MANIFEST = DATA / "manifest.csv"
SMOKE_MANIFEST = DATA / "manifest-smoke.csv"
LABELS = REPO / "eval" / "neon-labels.csv"
CLASSES = Path(__file__).resolve().parent / "classes.yaml"

USER_AGENT = "nahpu-ai research; https://github.com/nahpu/nahpu-ai"
REQUEST_GAP_S = 1.0
RETRIES = 4
LONG_SIDE = 512
SMOKE_PER_CLASS = 5
NEON = "NEON (National Ecological Observatory Network) Biorepository"

TAXON_FIELDS = ["kingdom", "phylum", "class", "order", "family", "genus", "scientificName"]
COLUMNS = [
    "path", "class", "draw_rank",
    "kingdom", "phylum", "taxon_class", "order", "family", "genus", "scientificName",
    "occid", "occurrenceID", "catalogNumber", "collid", "collectionCode", "view",
    "source_url", "source_sha256", "source_bytes",
    "sha256", "width", "height",
    "licence", "owner", "creator", "dwca_pubdate", "accessed_at", "citation",
]

csv.field_size_limit(sys.maxsize)


def log(*parts):
    print(*parts, flush=True)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(path.read_bytes())


class Fetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.last_request = 0.0
        self.requests_made = 0

    def get(self, url):
        for attempt in range(RETRIES):
            wait = self.last_request + REQUEST_GAP_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.monotonic()
            self.requests_made += 1
            try:
                response = self.session.get(url, timeout=120)
                if response.status_code == 200:
                    return response.content
                error = f"HTTP {response.status_code}"
                if response.status_code == 404:
                    break
            except requests.RequestException as exc:
                error = str(exc)
            backoff = 2 ** (attempt + 1)
            log(f"  retry {attempt + 1}/{RETRIES} in {backoff}s: {url} ({error})")
            time.sleep(backoff)
        raise RuntimeError(f"failed to fetch {url}: {error}")


def load_config():
    with open(CLASSES, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dwca_path(code):
    return CACHE / f"NEON-{code}_DwC-A.zip"


def ensure_dwca(config, fetcher, codes, refresh):
    CACHE.mkdir(parents=True, exist_ok=True)
    for code in codes:
        path = dwca_path(code)
        if path.exists() and not refresh:
            continue
        url = config["dwca_url"].format(code=code)
        log(f"downloading {url}")
        path.write_bytes(fetcher.get(url))


def read_csv_from_zip(archive, name):
    with archive.open(name) as raw:
        yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))


def read_dwca(code):
    """Return (collection info, occurrences by id, media rows by occurrence id)."""
    with zipfile.ZipFile(dwca_path(code)) as archive:
        eml = ET.fromstring(archive.read("eml.xml"))
        collection = eml.find(".//collection")
        info = {
            "code": code,
            "collid": collection.get("id") if collection is not None else "",
            "name": eml.findtext(".//dataset/title", default=""),
            "pubdate": eml.findtext(".//pubDate", default=""),
        }
        occurrences = {row["id"]: row for row in read_csv_from_zip(archive, "occurrences.csv")}
        media = defaultdict(list)
        for row in read_csv_from_zip(archive, "multimedia.csv"):
            media[row["coreid"]].append(row)
    return info, occurrences, media


def image_stem(url):
    stem = url.rsplit("/", 1)[-1]
    stem = re.sub(r"\.(jpe?g|png|tiff?)$", "", stem, flags=re.I)
    stem = re.sub(r"_(lg|med|tn)$", "", stem)
    return re.sub(r"_\d{10}$", "", stem)


def web_size_url(media_row):
    """Prefer the web-size copy (about 1400 px) over the large original."""
    urls = [media_row["accessURI"], media_row["goodQualityAccessURI"]]
    for url in urls:
        if url and not re.search(r"_(lg|med|tn)\.\w+$", url):
            return url
    return media_row["accessURI"]


def taxon_matches(occurrence, rules):
    include = rules.get("include", {})
    exclude = rules.get("exclude", {})
    if any(occurrence.get(field) not in values for field, values in include.items()):
        return False
    return not any(occurrence.get(field) in values for field, values in exclude.items())


def pick_view(media_rows, views, include_unlicensed):
    """Return (view label, media row) for the first view this specimen has."""
    usable = [m for m in media_rows if include_unlicensed or m["rights"].strip()]
    for view in views:
        pattern = re.compile(view["pattern"])
        hits = sorted(
            (m for m in usable if pattern.search(image_stem(m["accessURI"]))),
            key=lambda m: m["accessURI"],
        )
        if hits:
            return view["label"], hits[0]
    return None, None


def candidates_for(cls, archives, include_unlicensed):
    found = []
    for code in cls["collections"]:
        info, occurrences, media = archives[code]
        for occid, media_rows in media.items():
            occurrence = occurrences.get(occid)
            if occurrence is None or not taxon_matches(occurrence, cls):
                continue
            label, media_row = pick_view(media_rows, cls["views"], include_unlicensed)
            if media_row is not None:
                found.append((info, occurrence, label, media_row))
    found.sort(key=lambda c: (c[0]["code"], int(c[1]["id"])))
    return found


def species_key(occurrence):
    return occurrence["scientificName"] or occurrence["genus"] or occurrence["family"]


def draw_sample(candidates, per_class, cap_share, rng):
    """Round-robin over species in a seeded order, with a per-species cap.

    If the cap stops the class from filling, raise it one step at a time.
    Returns (picked in draw order, cap used).
    """
    by_species = defaultdict(list)
    for candidate in candidates:
        by_species[species_key(candidate[1])].append(candidate)
    species = sorted(by_species)
    rng.shuffle(species)
    for name in species:
        rng.shuffle(by_species[name])

    target = min(per_class, len(candidates))
    cap = max(1, int(per_class * cap_share))
    while True:
        picked = []
        for round_index in range(cap):
            for name in species:
                pool = by_species[name]
                if round_index < len(pool) and len(picked) < target:
                    picked.append(pool[round_index])
        if len(picked) >= target:
            return picked, cap
        cap += 1


def creator_for_citation(creator):
    parts = creator.split()
    if len(parts) == 2:
        return f"{parts[1]}, {parts[0][0]}."
    return creator


def citation(row):
    filename = row["source_url"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    lead = f"{creator_for_citation(row['creator'])}, " if row["creator"] else ""
    return (
        f"{lead}{NEON}. Accessed via {NEON} Sample Portal [{row['accessed_at']}]. "
        f"{filename}. Photograph. {row['source_url']}"
    )


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def build_manifest(config, args, fetcher):
    codes = sorted({code for cls in config["classes"] for code in cls["collections"]})
    ensure_dwca(config, fetcher, codes, args.refresh_dwca)
    archives = {code: read_dwca(code) for code in codes}

    rows = []
    for cls in config["classes"]:
        candidates = candidates_for(cls, archives, args.include_unlicensed)
        rng = random.Random(f"{args.seed}:{cls['name']}")
        picked, cap = draw_sample(candidates, args.per_class, args.cap, rng)
        spread = Counter(species_key(c[1]) for c in picked)
        note = "" if cap <= int(args.per_class * args.cap) else f" (cap relaxed to {cap})"
        log(
            f"{cls['name']}: {len(candidates)} candidates, "
            f"{len(Counter(species_key(c[1]) for c in candidates))} species; "
            f"picked {len(picked)} from {len(spread)} species, max {max(spread.values(), default=0)}"
            f" per species{note}"
        )
        if args.dry_run:
            for name, count in spread.most_common():
                log(f"    {count:3d}  {name}")
            continue
        for rank, (info, occurrence, label, media_row) in enumerate(picked):
            path = f"images/{cls['name']}/{safe_name(occurrence['catalogNumber'])}_{label}.jpg"
            row = {
                "path": path,
                "class": cls["name"],
                "draw_rank": rank,
                "taxon_class": occurrence["class"],
                "occid": occurrence["id"],
                "occurrenceID": occurrence["occurrenceID"],
                "catalogNumber": occurrence["catalogNumber"],
                "collid": info["collid"],
                "collectionCode": occurrence["collectionCode"] or info["code"],
                "view": label,
                "source_url": web_size_url(media_row),
                "licence": media_row["rights"].strip(),
                "owner": media_row["Owner"],
                "creator": media_row["creator"],
                "dwca_pubdate": info["pubdate"],
            }
            for field in TAXON_FIELDS:
                if field != "class":
                    row[field] = occurrence[field]
            rows.append(row)
    return rows


def read_manifest(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path, rows):
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in COLUMNS})
    tmp.replace(path)


def write_labels(config):
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["class", "common_name", "taxon_name"])
        for cls in config["classes"]:
            writer.writerow([cls["name"], cls["common_name"], cls["taxon_name"]])


def save_resized(source_bytes, path):
    with Image.open(io.BytesIO(source_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        scale = LONG_SIDE / max(image.size)
        if scale < 1:
            size = (round(image.width * scale), round(image.height * scale))
            image = image.resize(size, Image.Resampling.BICUBIC)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, "JPEG", quality=95)
        return image.size


def fetch_images(rows, fetcher, manifest_path):
    problems = []
    todo = [r for r in rows if not (r["sha256"] and (DATA / r["path"]).exists()
                                    and sha256_file(DATA / r["path"]) == r["sha256"])]
    log(f"{len(rows) - len(todo)} images already present, {len(todo)} to fetch")
    for index, row in enumerate(todo, 1):
        path = DATA / row["path"]
        try:
            data = fetcher.get(row["source_url"])
        except RuntimeError as exc:
            problems.append(str(exc))
            log(f"  {exc}")
            continue
        source_sha = sha256_bytes(data)
        if row["source_sha256"] and row["source_sha256"] != source_sha:
            problems.append(f"source changed upstream: {row['path']}")
            log(f"  source sha256 differs from manifest: {row['path']}")
            continue
        width, height = save_resized(data, path)
        saved_sha = sha256_file(path)
        if row["sha256"] and row["sha256"] != saved_sha:
            problems.append(f"saved file differs (Pillow version?): {row['path']}")
            log(f"  saved sha256 differs from manifest: {row['path']}")
            continue
        row.update(
            source_sha256=source_sha,
            source_bytes=len(data),
            sha256=saved_sha,
            width=width,
            height=height,
        )
        if not row["accessed_at"]:
            row["accessed_at"] = dt.date.today().isoformat()
        row["citation"] = citation(row)
        if index % 10 == 0 or index == len(todo):
            write_manifest(manifest_path, rows)
            log(f"  {index}/{len(todo)} fetched")
    write_manifest(manifest_path, rows)
    return problems


def verify(manifest_path):
    rows = read_manifest(manifest_path)
    bad = []
    for row in rows:
        path = DATA / row["path"]
        if not path.exists():
            bad.append(f"missing: {row['path']}")
        elif sha256_file(path) != row["sha256"]:
            bad.append(f"sha256 mismatch: {row['path']}")
        elif not row["licence"]:
            bad.append(f"no licence: {row['path']}")
    for line in bad:
        log(line)
    log(f"verified {len(rows) - len(bad)}/{len(rows)} images in {manifest_path.name}")
    return not bad


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--cap", type=float, default=0.25, help="max share of one species per class")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-unlicensed", action="store_true", help="keep images with empty rights")
    parser.add_argument("--resample", action="store_true", help="draw a new sample even if a manifest exists")
    parser.add_argument("--refresh-dwca", action="store_true", help="download the archives again")
    parser.add_argument("--dry-run", action="store_true", help="show the sample, fetch no images")
    parser.add_argument("--smoke", action="store_true", help="write and fetch the 5 per class subset")
    parser.add_argument("--verify", action="store_true", help="re-hash saved images against the manifest")
    args = parser.parse_args()

    if args.verify:
        ok = verify(MANIFEST)
        if SMOKE_MANIFEST.exists():
            ok = verify(SMOKE_MANIFEST) and ok
        sys.exit(0 if ok else 1)

    config = load_config()
    write_labels(config)
    fetcher = Fetcher()
    started = time.monotonic()

    if args.dry_run or args.resample or not MANIFEST.exists():
        rows = build_manifest(config, args, fetcher)
        if args.dry_run:
            return
        write_manifest(MANIFEST, rows)
        log(f"wrote {MANIFEST.relative_to(REPO)} with {len(rows)} rows")
    rows = read_manifest(MANIFEST)

    if args.smoke:
        smoke = [r for r in rows if int(r["draw_rank"]) < SMOKE_PER_CLASS]
        problems = fetch_images(smoke, fetcher, SMOKE_MANIFEST)
        by_path = {r["path"]: r for r in smoke}
        rows = [by_path.get(r["path"], r) for r in rows]
        write_manifest(MANIFEST, rows)
    else:
        problems = fetch_images(rows, fetcher, MANIFEST)

    log(f"done: {fetcher.requests_made} requests in {time.monotonic() - started:.0f}s")
    if problems:
        log(f"{len(problems)} problems:")
        for line in problems:
            log(f"  {line}")
        sys.exit(1)


if __name__ == "__main__":
    main()
