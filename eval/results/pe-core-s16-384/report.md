# Zero-shot evaluation: pe-core-s16-384

- Backend: burn-vulkan
- Manifest: `../data/neon/manifest.csv`, 700 images
- Model load 0.68 s; per image (decode + preprocess + embed) median 100 ms, p95 104 ms

## All settings, best macro accuracy first

| Framing | Prompts | Wording | Top-1 | Top-3 | Macro |
|---|---|---|---|---|---|
| centre-crop | specimen | both | 78.9% | 98.6% | 78.9% |
| centre-crop | specimen | common | 78.0% | 97.6% | 78.0% |
| centre-crop | photo | both | 77.7% | 97.3% | 77.7% |
| centre-crop | ensemble | both | 77.3% | 97.9% | 77.3% |
| pad | photo | both | 75.4% | 97.6% | 75.4% |
| pad | bare | both | 75.0% | 97.7% | 75.0% |
| centre-crop | bare | both | 74.7% | 98.1% | 74.7% |
| pad | specimen | both | 74.4% | 96.4% | 74.4% |
| centre-crop | bare | common | 74.4% | 98.6% | 74.4% |
| centre-crop | ensemble | common | 74.1% | 96.7% | 74.1% |
| pad | ensemble | both | 74.0% | 97.0% | 74.0% |
| centre-crop | photo | common | 73.9% | 95.7% | 73.9% |
| pad | bare | common | 71.7% | 96.3% | 71.7% |
| pad | specimen | common | 70.6% | 94.6% | 70.6% |
| pad | ensemble | common | 69.0% | 94.9% | 69.0% |
| pad | photo | common | 68.6% | 94.9% | 68.6% |
| pad | photo | taxon | 56.1% | 84.1% | 56.1% |
| pad | ensemble | taxon | 54.7% | 84.9% | 54.7% |
| pad | specimen | taxon | 54.4% | 78.4% | 54.4% |
| centre-crop | photo | taxon | 53.1% | 80.7% | 53.1% |
| pad | bare | taxon | 51.6% | 81.1% | 51.6% |
| centre-crop | ensemble | taxon | 51.4% | 83.4% | 51.4% |
| centre-crop | specimen | taxon | 50.0% | 78.1% | 50.0% |
| centre-crop | bare | taxon | 45.6% | 80.4% | 45.6% |

## Default setting (centre-crop, ensemble, common): centre-crop / ensemble / common

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 100.0% |
| mouse_or_vole | 100 | 35.0% |
| frog_or_toad | 100 | 39.0% |
| salamander | 100 | 64.0% |
| lizard | 100 | 81.0% |
| ground_beetle | 100 | 100.0% |
| mosquito | 100 | 100.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 65 | 35 | 0 | 0 | 0 | 0 | 0 |
| **frog_or_toad** | 15 | 0 | 39 | 1 | 21 | 24 | 0 |
| **salamander** | 2 | 0 | 0 | 64 | 33 | 1 | 0 |
| **lizard** | 4 | 0 | 3 | 9 | 81 | 3 | 0 |
| **ground_beetle** | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| **mosquito** | 0 | 0 | 0 | 0 | 0 | 0 | 100 |

## Best setting: centre-crop / specimen / both

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 99.0% |
| mouse_or_vole | 100 | 51.0% |
| frog_or_toad | 100 | 53.0% |
| salamander | 100 | 85.0% |
| lizard | 100 | 64.0% |
| ground_beetle | 100 | 100.0% |
| mosquito | 100 | 100.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 99 | 1 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 49 | 51 | 0 | 0 | 0 | 0 | 0 |
| **frog_or_toad** | 15 | 1 | 53 | 2 | 12 | 17 | 0 |
| **salamander** | 1 | 0 | 0 | 85 | 13 | 1 | 0 |
| **lizard** | 5 | 6 | 6 | 18 | 64 | 1 | 0 |
| **ground_beetle** | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| **mosquito** | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
