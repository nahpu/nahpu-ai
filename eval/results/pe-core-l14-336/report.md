# Zero-shot evaluation: pe-core-l14-336

- Backend: burn-vulkan
- Manifest: `../data/neon/manifest.csv`, 700 images
- Model load 2.13 s; per image (decode + preprocess + embed) median 191 ms, p95 199 ms

## All settings, best macro accuracy first

| Framing | Prompts | Wording | Top-1 | Top-3 | Macro |
|---|---|---|---|---|---|
| centre-crop | specimen | common | 94.6% | 100.0% | 94.6% |
| centre-crop | ensemble | common | 94.3% | 99.9% | 94.3% |
| pad | specimen | common | 94.0% | 100.0% | 94.0% |
| centre-crop | bare | common | 93.6% | 99.7% | 93.6% |
| centre-crop | ensemble | both | 93.6% | 99.9% | 93.6% |
| centre-crop | specimen | both | 93.4% | 99.9% | 93.4% |
| pad | ensemble | common | 93.4% | 99.9% | 93.4% |
| centre-crop | photo | common | 93.0% | 99.9% | 93.0% |
| pad | ensemble | both | 93.0% | 99.4% | 93.0% |
| pad | specimen | both | 92.7% | 99.6% | 92.7% |
| pad | bare | common | 92.4% | 99.7% | 92.4% |
| pad | photo | common | 92.3% | 99.9% | 92.3% |
| centre-crop | photo | both | 91.1% | 99.9% | 91.1% |
| pad | photo | both | 90.1% | 99.4% | 90.1% |
| centre-crop | bare | both | 89.0% | 99.9% | 89.0% |
| pad | bare | both | 87.6% | 99.4% | 87.6% |
| centre-crop | specimen | taxon | 81.1% | 86.0% | 81.1% |
| centre-crop | photo | taxon | 80.9% | 94.4% | 80.9% |
| centre-crop | ensemble | taxon | 80.4% | 95.7% | 80.4% |
| pad | specimen | taxon | 78.0% | 85.1% | 78.0% |
| pad | ensemble | taxon | 77.4% | 93.4% | 77.4% |
| pad | photo | taxon | 76.7% | 91.9% | 76.7% |
| centre-crop | bare | taxon | 74.0% | 97.6% | 74.0% |
| pad | bare | taxon | 68.6% | 96.0% | 68.6% |

## Default setting (centre-crop, ensemble, common): centre-crop / ensemble / common

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 93.0% |
| mouse_or_vole | 100 | 96.0% |
| frog_or_toad | 100 | 100.0% |
| salamander | 100 | 100.0% |
| lizard | 100 | 71.0% |
| ground_beetle | 100 | 100.0% |
| mosquito | 100 | 100.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 93 | 7 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 4 | 96 | 0 | 0 | 0 | 0 | 0 |
| **frog_or_toad** | 0 | 0 | 100 | 0 | 0 | 0 | 0 |
| **salamander** | 0 | 0 | 0 | 100 | 0 | 0 | 0 |
| **lizard** | 0 | 0 | 1 | 28 | 71 | 0 | 0 |
| **ground_beetle** | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| **mosquito** | 0 | 0 | 0 | 0 | 0 | 0 | 100 |

## Best setting: centre-crop / specimen / common

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 91.0% |
| mouse_or_vole | 100 | 100.0% |
| frog_or_toad | 100 | 100.0% |
| salamander | 100 | 100.0% |
| lizard | 100 | 71.0% |
| ground_beetle | 100 | 100.0% |
| mosquito | 100 | 100.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 91 | 9 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 0 | 100 | 0 | 0 | 0 | 0 | 0 |
| **frog_or_toad** | 0 | 0 | 100 | 0 | 0 | 0 | 0 |
| **salamander** | 0 | 0 | 0 | 100 | 0 | 0 | 0 |
| **lizard** | 0 | 0 | 1 | 28 | 71 | 0 | 0 |
| **ground_beetle** | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| **mosquito** | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
