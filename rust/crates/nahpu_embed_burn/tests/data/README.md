# Test reference data

- `candle-reference.tsv`: MiniLM vectors from the Candle engine in rust-inference-bench.
- `jina-*-reference.tsv` and `*.pixels.rgb`: written by
  `scripts/reference/make_jina_fixtures.py` (onnxruntime on the same ONNX files).
  `*.pixels.rgb` is the 224x224 RGB crop PIL produces, raw bytes, row-major.

The two fixture photos are fetched by `scripts/fetch-model.sh`, not committed.
The crops here are derived from them and keep their licence (CC BY-SA 4.0):

- `fixture-neon-shrew`: *Sorex cinereus*, NEON-MAMC-VSS, NEON (National Ecological
  Observatory Network) Biorepository.
  https://biorepo.neonscience.org/media/NEON_MAMC-VSS/00000/D00000153413-1_1701716781_lg.jpg
- `fixture-neon-beetle`: *Pasimachus californicus*, NEON-CARC-AP, University of
  Florida Gainesville, via the NEON Biorepository.
  https://biorepo.neonscience.org/media/NEON_CARC-AP/OAES_/OAES_009.20160727.PASCAL.01.8_dorsal.jpg
