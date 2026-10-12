# Zero-shot evaluation: jina-clip-v1

- Backend: burn-metal
- Manifest: `../data/neon/manifest.csv`, 700 images
- Model load 1.36 s; per image (decode + preprocess + embed) median 42 ms, p95 49 ms

## All settings, best macro accuracy first

| Framing | Prompts | Wording | Top-1 | Top-3 | Macro |
|---|---|---|---|---|---|
| centre-crop | ensemble | common | 66.1% | 95.9% | 66.1% |
| centre-crop | photo | common | 66.0% | 94.3% | 66.0% |
| centre-crop | photo | both | 64.6% | 95.3% | 64.6% |
| centre-crop | specimen | common | 64.3% | 94.4% | 64.3% |
| centre-crop | bare | both | 64.0% | 95.3% | 64.0% |
| centre-crop | bare | common | 63.6% | 94.7% | 63.6% |
| centre-crop | ensemble | both | 63.4% | 95.4% | 63.4% |
| centre-crop | specimen | both | 62.3% | 95.6% | 62.3% |
| pad | bare | both | 58.1% | 94.0% | 58.1% |
| pad | photo | both | 57.7% | 94.0% | 57.7% |
| pad | ensemble | both | 57.4% | 92.7% | 57.4% |
| pad | ensemble | common | 57.3% | 90.9% | 57.3% |
| pad | specimen | both | 57.0% | 88.3% | 57.0% |
| pad | photo | common | 57.0% | 90.7% | 57.0% |
| pad | bare | common | 56.9% | 92.0% | 56.9% |
| pad | specimen | common | 55.6% | 87.9% | 55.6% |
| pad | bare | taxon | 39.4% | 71.0% | 39.4% |
| pad | photo | taxon | 37.0% | 69.6% | 37.0% |
| pad | specimen | taxon | 37.0% | 69.9% | 37.0% |
| pad | ensemble | taxon | 37.0% | 70.0% | 37.0% |
| centre-crop | bare | taxon | 35.9% | 66.7% | 35.9% |
| centre-crop | photo | taxon | 35.0% | 64.6% | 35.0% |
| centre-crop | specimen | taxon | 33.9% | 68.4% | 33.9% |
| centre-crop | ensemble | taxon | 33.4% | 66.7% | 33.4% |

## Default setting (centre-crop, ensemble, common): centre-crop / ensemble / common

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 0.0% |
| mouse_or_vole | 100 | 95.0% |
| frog_or_toad | 100 | 56.0% |
| salamander | 100 | 46.0% |
| lizard | 100 | 67.0% |
| ground_beetle | 100 | 99.0% |
| mosquito | 100 | 100.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 0 | 100 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 4 | 95 | 0 | 0 | 0 | 0 | 1 |
| **frog_or_toad** | 0 | 4 | 56 | 7 | 18 | 12 | 3 |
| **salamander** | 1 | 1 | 0 | 46 | 52 | 0 | 0 |
| **lizard** | 0 | 4 | 7 | 14 | 67 | 2 | 6 |
| **ground_beetle** | 0 | 0 | 0 | 0 | 0 | 99 | 1 |
| **mosquito** | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
