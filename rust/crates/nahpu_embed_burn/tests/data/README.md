# Test reference data

- `candle-reference.tsv`: MiniLM vectors from the Candle engine in rust-inference-bench.
- `jina-*-reference.tsv` and `*.pixels.rgb`: written by
  `scripts/reference/make_jina_fixtures.py` (onnxruntime on the same ONNX files).
  `*.pixels.rgb` is the 224x224 RGB crop PIL produces, raw bytes, row-major.
- `bioclip-*-reference.tsv`: written by `scripts/bioclip/export_onnx.py` from
  open_clip in PyTorch. `bioclip-photo-reference.tsv` uses open_clip's own
  preprocessing on the photo file.
- `pe-core-*-reference.tsv`: written by `scripts/pe/export_onnx.py` from
  open_clip in PyTorch, one set for each size (`t`, `s`, `b`, `l`, `g`).
  `pe-core-*-image-reference.tsv` uses the crop in `*.pixels.rgb` centred on a
  square of grey 128 of the size's own width (224 to 448), so no larger crop is
  committed. `pe-core-*-photo-reference.tsv` resizes the photo file as
  `preprocess.rs` does (bicubic, centre crop), not as the model's own config
  does (bilinear, no crop).
- `nomic-*-reference.tsv`: written by `scripts/nomic/make_references.py`
  (onnxruntime on the same ONNX files; the photos are resized with Pillow).

The two fixture photos are fetched by `scripts/fetch-model.sh`, not committed.
The crops here are derived from them and keep their licence (CC BY-SA 4.0):

- `fixture-neon-shrew`: *Sorex cinereus*, NEON-MAMC-VSS, NEON (National Ecological
  Observatory Network) Biorepository.
  https://biorepo.neonscience.org/media/NEON_MAMC-VSS/00000/D00000153413-1_1701716781_lg.jpg
- `fixture-neon-beetle`: *Pasimachus californicus*, NEON-CARC-AP, University of
  Florida Gainesville, via the NEON Biorepository.
  https://biorepo.neonscience.org/media/NEON_CARC-AP/OAES_/OAES_009.20160727.PASCAL.01.8_dorsal.jpg

The six Nomic photos are fetched the same way, under the names below. The
vectors in `nomic-vision-image-reference.tsv` are derived from them and keep
their licence (CC BY-SA 4.0). All are from the NEON Biorepository, accessed
2026-10-07:

- `emb01_beetle_chlaenius_aestivus_dorsal`: *Chlaenius aestivus*, NEON-CARC-PV,
  https://doi.org/10.58052/NEON033RR
  https://biorepo.neonscience.org/media/NEON_CARC-PV/NEONBET/NEON.BET.D08.002076_dorsal_1x.tiff.jpg
- `emb05_mosquito_aedes_monticola_lateral`: *Aedes monticola*, NEON-MOSC-PV,
  https://doi.org/10.58052/NEON003X7
  https://biorepo.neonscience.org/media/NEON_MOSC-PV/MOS/MOS.D14.000124_lateral_2x_med.jpg
- `emb06_lizard_holbrookia_maculata_dorsal`: *Holbrookia maculata*,
  NEON-HEVC-GBTS, https://doi.org/10.58052/NEON09U2R
  https://biorepo.neonscience.org/media/NEON_HEVC-GBTS/A00000255/A00000255702-1_med.jpg
- `emb07_salamander_plethodon_cinereus_three`: *Plethodon cinereus*,
  NEON-HEVC-GBAP, https://doi.org/10.58052/NEON00WWF
  https://biorepo.neonscience.org/media/NEON_HEVC-GBAP/A00000328/A00000328482-2.jpg
- `emb08_toad_anaxyrus_americanus_two`: *Anaxyrus americanus*, NEON-HEVC-GBAP,
  https://doi.org/10.58052/NEON07E7J
  https://biorepo.neonscience.org/media/NEON_HEVC-GBAP/00000/A00000360404-1_1713917340.jpg
- `emb10_mammal_peromyscus_ventral`: *Peromyscus sonoriensis*, NEON-MAMC-VSS,
  https://doi.org/10.58052/NEON048GN
  https://biorepo.neonscience.org/media/NEON_MAMC-VSS/00000/D00000122950-3_1663177565.jpg
