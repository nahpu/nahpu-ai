# Zero-shot evaluation: bioclip

- Backend: burn-metal
- Manifest: `../data/neon/manifest.csv`, 700 images
- Model load 0.85 s; per image (decode + preprocess + embed) median 38 ms, p95 47 ms

## All settings, best macro accuracy first

| Framing | Prompts | Wording | Top-1 | Top-3 | Macro |
|---|---|---|---|---|---|
| centre-crop | specimen | common | 86.9% | 99.0% | 86.9% |
| centre-crop | specimen | both | 81.9% | 95.3% | 81.9% |
| centre-crop | ensemble | common | 80.7% | 97.3% | 80.7% |
| pad | specimen | common | 80.4% | 97.1% | 80.4% |
| centre-crop | bare | common | 77.7% | 86.0% | 77.7% |
| centre-crop | bare | both | 77.1% | 95.1% | 77.1% |
| centre-crop | photo | common | 76.1% | 94.4% | 76.1% |
| centre-crop | ensemble | both | 76.1% | 96.6% | 76.1% |
| centre-crop | ensemble | taxon | 75.9% | 95.6% | 75.9% |
| pad | bare | both | 72.9% | 93.9% | 72.9% |
| pad | specimen | both | 71.6% | 86.0% | 71.6% |
| centre-crop | photo | both | 71.4% | 94.9% | 71.4% |
| pad | ensemble | taxon | 71.0% | 92.1% | 71.0% |
| pad | ensemble | common | 69.1% | 91.3% | 69.1% |
| centre-crop | photo | taxon | 69.0% | 90.9% | 69.0% |
| pad | ensemble | both | 67.9% | 89.3% | 67.9% |
| pad | photo | common | 67.4% | 85.9% | 67.4% |
| pad | photo | both | 66.3% | 87.0% | 66.3% |
| pad | photo | taxon | 63.3% | 87.0% | 63.3% |
| pad | bare | common | 63.0% | 83.7% | 63.0% |
| centre-crop | specimen | taxon | 54.4% | 79.6% | 54.4% |
| centre-crop | bare | taxon | 54.3% | 82.3% | 54.3% |
| pad | bare | taxon | 50.6% | 80.0% | 50.6% |
| pad | specimen | taxon | 49.3% | 74.0% | 49.3% |

## Default setting (centre-crop, ensemble, common): centre-crop / ensemble / common

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 100.0% |
| mouse_or_vole | 100 | 26.0% |
| frog_or_toad | 100 | 61.0% |
| salamander | 100 | 86.0% |
| lizard | 100 | 93.0% |
| ground_beetle | 100 | 99.0% |
| mosquito | 100 | 100.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 71 | 26 | 0 | 0 | 3 | 0 | 0 |
| **frog_or_toad** | 0 | 0 | 61 | 10 | 26 | 2 | 1 |
| **salamander** | 0 | 0 | 0 | 86 | 14 | 0 | 0 |
| **lizard** | 0 | 0 | 2 | 5 | 93 | 0 | 0 |
| **ground_beetle** | 0 | 0 | 0 | 0 | 1 | 99 | 0 |
| **mosquito** | 0 | 0 | 0 | 0 | 0 | 0 | 100 |

## Best setting: centre-crop / specimen / common

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 95.0% |
| mouse_or_vole | 100 | 76.0% |
| frog_or_toad | 100 | 81.0% |
| salamander | 100 | 95.0% |
| lizard | 100 | 75.0% |
| ground_beetle | 100 | 86.0% |
| mosquito | 100 | 100.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 95 | 5 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 17 | 76 | 2 | 0 | 5 | 0 | 0 |
| **frog_or_toad** | 0 | 0 | 81 | 9 | 6 | 3 | 1 |
| **salamander** | 0 | 0 | 4 | 95 | 1 | 0 | 0 |
| **lizard** | 0 | 0 | 5 | 17 | 75 | 0 | 3 |
| **ground_beetle** | 1 | 0 | 1 | 1 | 9 | 86 | 2 |
| **mosquito** | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
