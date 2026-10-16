# NEON specimen-photo test set

A small, fixed set of museum specimen photos for the zero-shot classification
eval. It has 7 classes and about 100 specimens per class, one photo each:

| Class | Source collection | Taxon | View |
|---|---|---|---|
| shrew | MAMC-VSS (collid 28) | Soricidae | `-1` whole body |
| mouse_or_vole | MAMC-VSS (collid 28) | Rodentia, minus squirrels and gophers | `-1` whole body |
| frog_or_toad | HEVC-GBTS (15), HEVC-GBAP (12) | Anura | `-1` dorsal |
| salamander | HEVC-GBTS (15), HEVC-GBAP (12) | Caudata | `-1` dorsal |
| lizard | HEVC-GBTS (15), HEVC-GBAP (12) | Squamata, minus snakes | `-1` dorsal |
| ground_beetle | CARC-PV (39) | Carabidae | dorsal |
| mosquito | MOSC-PV (29) | Culicidae | lateral, else dorsal |

The photos come from the
[NEON Biorepository](https://biorepo.neonscience.org/portal/). The class
rules live in `scripts/neon/classes.yaml`. The labels the classifier reads are
in `eval/neon-labels.csv`.

## What is committed

- `manifest.csv`: one row per photo. It holds the taxonomy, NEON IDs, source
  URL, sha256 of the source and saved files, size, licence, owner, creator and
  a citation. It works as a lock file.
- `manifest-smoke.csv`: 5 photos per class, taken from the full manifest.

The images (`images/`) and the archive cache (`cache/`) are not committed.

## Fetch and verify

```sh
python3 -m venv scripts/neon/.venv
scripts/neon/.venv/bin/pip install -r scripts/neon/requirements.txt
scripts/neon/.venv/bin/python -I scripts/neon/fetch_dataset.py           # all images
scripts/neon/.venv/bin/python -I scripts/neon/fetch_dataset.py --smoke   # 5 per class only
scripts/neon/.venv/bin/python -I scripts/neon/fetch_dataset.py --verify  # re-hash
```

The script fetches the images listed in `manifest.csv`, one at a time with a
1 second gap. A full fetch takes about 14 minutes and uses about 33 MB. Files that already match
the manifest are skipped. `--verify` exits non-zero if any file is missing or
its hash differs.

Each photo is saved with EXIF rotation applied, as RGB, resized so the long
side is 512 px (PIL bicubic, no upscaling), as JPEG quality 95 with no EXIF.
The pinned Pillow version matters: a different version can change the saved
bytes.

To draw a new sample from the latest archives, run with `--resample`. Add
`--refresh-dwca` to download the archives again (NEON rebuilds them weekly).
Sampling is seeded (`--seed 0`) and spreads picks across species, with no
species above 25% of a class.

## Licence

- Images: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  Photos with an empty `rights` field are left out by default.
- Specimen records: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The resized copies are adapted works, so they stay under CC BY-SA 4.0.

## Citation

Each manifest row has a `citation` built from the NEON image template, for
example:

> Steger, L., NEON (National Ecological Observatory Network) Biorepository.
> Accessed via NEON (National Ecological Observatory Network) Biorepository
> Sample Portal [2026-10-12]. B00000021553_1611185823. Photograph.
> https://biorepo.neonscience.org/media/NEON_MAMC-VSS/00000/B00000021553_1611185823.jpg

Cite the record data per collection, as NEON asks:

> NEON (National Ecological Observatory Network) Biorepository. [Collection
> Name]. Data accessed from
> https://biorepo.neonscience.org/portal/collections/misc/neoncollprofiles.php?collid=[collid]
> on [date]. Licensed under CC BY 4.0.

See https://biorepo.neonscience.org/portal/misc/cite.php for details.
