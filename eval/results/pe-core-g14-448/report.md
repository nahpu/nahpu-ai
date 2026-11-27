# Zero-shot evaluation: pe-core-g14-448

- Backend: burn-vulkan
- Manifest: `../data/neon/manifest.csv`, 700 images
- Model load 13.04 s; per image (decode + preprocess + embed) median 538 ms, p95 803 ms

## All settings, best macro accuracy first

| Framing | Prompts | Wording | Top-1 | Top-3 | Macro |
|---|---|---|---|---|---|
| centre-crop | ensemble | common | 97.4% | 100.0% | 97.4% |
| centre-crop | specimen | common | 97.1% | 100.0% | 97.1% |
| centre-crop | ensemble | both | 97.1% | 100.0% | 97.1% |
| centre-crop | bare | both | 96.9% | 100.0% | 96.9% |
| pad | ensemble | both | 96.9% | 100.0% | 96.9% |
| pad | ensemble | common | 96.6% | 99.9% | 96.6% |
| centre-crop | specimen | both | 96.3% | 100.0% | 96.3% |
| pad | bare | both | 96.1% | 99.9% | 96.1% |
| pad | specimen | common | 96.1% | 99.9% | 96.1% |
| pad | specimen | both | 95.7% | 99.9% | 95.7% |
| pad | photo | both | 95.0% | 100.0% | 95.0% |
| centre-crop | bare | common | 94.6% | 100.0% | 94.6% |
| pad | photo | common | 94.6% | 99.9% | 94.6% |
| pad | bare | common | 94.4% | 99.9% | 94.4% |
| centre-crop | photo | common | 94.3% | 100.0% | 94.3% |
| centre-crop | photo | both | 94.3% | 100.0% | 94.3% |
| centre-crop | specimen | taxon | 78.0% | 97.6% | 78.0% |
| pad | specimen | taxon | 77.7% | 98.7% | 77.7% |
| pad | ensemble | taxon | 77.0% | 94.3% | 77.0% |
| centre-crop | ensemble | taxon | 76.9% | 90.6% | 76.9% |
| pad | photo | taxon | 75.6% | 91.9% | 75.6% |
| centre-crop | photo | taxon | 75.3% | 91.3% | 75.3% |
| pad | bare | taxon | 75.1% | 97.6% | 75.1% |
| centre-crop | bare | taxon | 74.6% | 97.4% | 74.6% |

## Default setting (centre-crop, ensemble, common): centre-crop / ensemble / common

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 99.0% |
| mouse_or_vole | 100 | 93.0% |
| frog_or_toad | 100 | 100.0% |
| salamander | 100 | 100.0% |
| lizard | 100 | 90.0% |
| ground_beetle | 100 | 100.0% |
| mosquito | 100 | 100.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 99 | 1 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 7 | 93 | 0 | 0 | 0 | 0 | 0 |
| **frog_or_toad** | 0 | 0 | 100 | 0 | 0 | 0 | 0 |
| **salamander** | 0 | 0 | 0 | 100 | 0 | 0 | 0 |
| **lizard** | 0 | 0 | 0 | 10 | 90 | 0 | 0 |
| **ground_beetle** | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| **mosquito** | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
