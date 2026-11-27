# Zero-shot evaluation: pe-core-t16-384

- Backend: burn-vulkan
- Manifest: `../data/neon/manifest.csv`, 700 images
- Model load 1.05 s; per image (decode + preprocess + embed) median 104 ms, p95 110 ms

## All settings, best macro accuracy first

| Framing | Prompts | Wording | Top-1 | Top-3 | Macro |
|---|---|---|---|---|---|
| centre-crop | specimen | both | 71.0% | 93.7% | 71.0% |
| pad | specimen | both | 70.3% | 91.6% | 70.3% |
| centre-crop | ensemble | both | 69.4% | 95.1% | 69.4% |
| centre-crop | specimen | common | 68.7% | 93.0% | 68.7% |
| pad | specimen | common | 67.9% | 90.0% | 67.9% |
| centre-crop | photo | both | 67.7% | 95.9% | 67.7% |
| centre-crop | ensemble | common | 67.7% | 93.3% | 67.7% |
| pad | ensemble | both | 67.7% | 91.3% | 67.7% |
| pad | photo | both | 66.4% | 93.9% | 66.4% |
| centre-crop | photo | common | 64.7% | 91.9% | 64.7% |
| pad | ensemble | common | 64.0% | 90.4% | 64.0% |
| pad | bare | common | 62.9% | 86.6% | 62.9% |
| centre-crop | bare | common | 62.1% | 87.7% | 62.1% |
| centre-crop | bare | both | 61.0% | 91.3% | 61.0% |
| pad | photo | common | 60.6% | 90.3% | 60.6% |
| pad | bare | both | 59.0% | 91.9% | 59.0% |
| pad | bare | taxon | 46.4% | 67.7% | 46.4% |
| pad | photo | taxon | 43.3% | 68.0% | 43.3% |
| pad | specimen | taxon | 42.4% | 69.4% | 42.4% |
| pad | ensemble | taxon | 41.6% | 66.3% | 41.6% |
| centre-crop | bare | taxon | 39.0% | 57.0% | 39.0% |
| centre-crop | specimen | taxon | 36.6% | 65.4% | 36.6% |
| centre-crop | ensemble | taxon | 35.7% | 58.0% | 35.7% |
| centre-crop | photo | taxon | 34.6% | 64.1% | 34.6% |

## Default setting (centre-crop, ensemble, common): centre-crop / ensemble / common

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 100.0% |
| mouse_or_vole | 100 | 18.0% |
| frog_or_toad | 100 | 16.0% |
| salamander | 100 | 56.0% |
| lizard | 100 | 84.0% |
| ground_beetle | 100 | 100.0% |
| mosquito | 100 | 100.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 82 | 18 | 0 | 0 | 0 | 0 | 0 |
| **frog_or_toad** | 31 | 0 | 16 | 7 | 30 | 15 | 1 |
| **salamander** | 3 | 0 | 0 | 56 | 41 | 0 | 0 |
| **lizard** | 10 | 0 | 3 | 3 | 84 | 0 | 0 |
| **ground_beetle** | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| **mosquito** | 0 | 0 | 0 | 0 | 0 | 0 | 100 |

## Best setting: centre-crop / specimen / both

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 99.0% |
| mouse_or_vole | 100 | 28.0% |
| frog_or_toad | 100 | 26.0% |
| salamander | 100 | 85.0% |
| lizard | 100 | 60.0% |
| ground_beetle | 100 | 99.0% |
| mosquito | 100 | 100.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 99 | 1 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 72 | 28 | 0 | 0 | 0 | 0 | 0 |
| **frog_or_toad** | 12 | 3 | 26 | 20 | 22 | 15 | 2 |
| **salamander** | 2 | 0 | 0 | 85 | 13 | 0 | 0 |
| **lizard** | 3 | 0 | 5 | 32 | 60 | 0 | 0 |
| **ground_beetle** | 0 | 0 | 0 | 0 | 0 | 99 | 1 |
| **mosquito** | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
