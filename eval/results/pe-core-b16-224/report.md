# Zero-shot evaluation: pe-core-b16-224

- Backend: burn-vulkan
- Manifest: `../data/neon/manifest.csv`, 700 images
- Model load 1.70 s; per image (decode + preprocess + embed) median 64 ms, p95 69 ms

## All settings, best macro accuracy first

| Framing | Prompts | Wording | Top-1 | Top-3 | Macro |
|---|---|---|---|---|---|
| centre-crop | specimen | both | 88.7% | 99.7% | 88.7% |
| centre-crop | bare | both | 88.0% | 98.9% | 88.0% |
| centre-crop | ensemble | both | 87.7% | 98.9% | 87.7% |
| centre-crop | specimen | common | 87.1% | 99.7% | 87.1% |
| centre-crop | ensemble | common | 87.0% | 99.7% | 87.0% |
| centre-crop | photo | common | 86.9% | 99.3% | 86.9% |
| centre-crop | photo | both | 86.6% | 98.7% | 86.6% |
| pad | specimen | both | 86.3% | 98.7% | 86.3% |
| pad | specimen | common | 86.0% | 98.7% | 86.0% |
| centre-crop | bare | common | 85.9% | 99.4% | 85.9% |
| pad | ensemble | both | 84.3% | 98.3% | 84.3% |
| pad | photo | both | 84.0% | 97.4% | 84.0% |
| pad | ensemble | common | 83.6% | 98.0% | 83.6% |
| pad | bare | common | 83.4% | 97.9% | 83.4% |
| pad | bare | both | 83.4% | 96.9% | 83.4% |
| pad | photo | common | 83.1% | 97.7% | 83.1% |
| centre-crop | photo | taxon | 56.9% | 88.0% | 56.9% |
| centre-crop | ensemble | taxon | 55.6% | 88.9% | 55.6% |
| pad | ensemble | taxon | 55.6% | 87.0% | 55.6% |
| pad | photo | taxon | 55.3% | 86.6% | 55.3% |
| centre-crop | bare | taxon | 54.6% | 86.7% | 54.6% |
| pad | bare | taxon | 52.1% | 85.0% | 52.1% |
| centre-crop | specimen | taxon | 51.0% | 85.1% | 51.0% |
| pad | specimen | taxon | 50.4% | 84.0% | 50.4% |

## Default setting (centre-crop, ensemble, common): centre-crop / ensemble / common

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 96.0% |
| mouse_or_vole | 100 | 75.0% |
| frog_or_toad | 100 | 91.0% |
| salamander | 100 | 79.0% |
| lizard | 100 | 69.0% |
| ground_beetle | 100 | 100.0% |
| mosquito | 100 | 99.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 96 | 4 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 25 | 75 | 0 | 0 | 0 | 0 | 0 |
| **frog_or_toad** | 1 | 0 | 91 | 0 | 1 | 7 | 0 |
| **salamander** | 0 | 0 | 0 | 79 | 21 | 0 | 0 |
| **lizard** | 7 | 0 | 8 | 13 | 69 | 3 | 0 |
| **ground_beetle** | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| **mosquito** | 0 | 0 | 0 | 0 | 0 | 1 | 99 |

## Best setting: centre-crop / specimen / both

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 90.0% |
| mouse_or_vole | 100 | 93.0% |
| frog_or_toad | 100 | 95.0% |
| salamander | 100 | 79.0% |
| lizard | 100 | 65.0% |
| ground_beetle | 100 | 100.0% |
| mosquito | 100 | 99.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 90 | 10 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 7 | 93 | 0 | 0 | 0 | 0 | 0 |
| **frog_or_toad** | 1 | 0 | 95 | 0 | 1 | 3 | 0 |
| **salamander** | 0 | 0 | 0 | 79 | 21 | 0 | 0 |
| **lizard** | 7 | 0 | 9 | 18 | 65 | 1 | 0 |
| **ground_beetle** | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| **mosquito** | 0 | 0 | 0 | 0 | 0 | 1 | 99 |
