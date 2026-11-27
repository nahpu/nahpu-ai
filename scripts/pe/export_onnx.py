"""Exports a PE-Core size's two towers to ONNX and writes reference data for the Rust gates.

    scripts/pe/.venv/bin/python -I scripts/pe/export_onnx.py [NAME ...]

NAME is one of the names in pe_core.py, e.g. pe-core-b16-224. With none,
exports pe-core-l14-336.

Writes into models/:
    NAME-text.onnx, NAME-vision.onnx, NAME.tokenizer.json
Writes into rust/crates/nahpu_embed_burn/tests/data/, with the size's letter for L:
    pe-core-l-text-reference.tsv, pe-core-l-image-reference.tsv, pe-core-l-photo-reference.tsv

The recipe is scripts/bioclip/export_onnx.py's: fixed shapes (batch 1), folded
by onnxruntime's basic optimiser, and references from open_clip in PyTorch
rather than from the exported ONNX, so the gates check the export as well as Burn.

G is exported in half precision: in single precision its image tower is 7.6 GB,
more than an 8 GB card holds. A tower still over 2 GB (G's image tower, 3.8 GB)
keeps its weights in NAME-vision.onnx.data beside the .onnx, since one ONNX file
holds 2 GB at most.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_core import CHECKPOINTS, FRAMINGS, HALF_PRECISION, PeCore, normalise_to_chw, upright_rgb

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
TEST_DATA = ROOT / "rust/crates/nahpu_embed_burn/tests/data"
DEFAULT = "pe-core-l14-336"
OPSET = 17
FIXTURE_CROP = 224
ONNX_FILE_LIMIT = 2 * 1024**3
# In half precision Burn's layer normalisation returns NaN for G, in both towers.
SINGLE_PRECISION_OPERATIONS = ["LayerNormalization"]

PROMPTS = [
    "a photo of a mouse.",
    "a photo of a museum specimen of a shrew.",
    "salamander",
]
FIXTURES = ["fixture-neon-shrew", "fixture-neon-beetle"]


class TextTower(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model.encode_text(input_ids)


class VisionTower(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model.encode_image(pixel_values)


def export(clip):
    text = TextTower(clip.model), clip.token_ids(PROMPTS[:1]), "input_ids", "text_embeds"
    vision = VisionTower(clip.model), torch.zeros(1, 3, clip.size, clip.size), "pixel_values", "image_embeds"
    for tower, (module, example, input_name, output_name) in [("text", text), ("vision", vision)]:
        # A scratch folder per tower: over 2 GB, the exporter leaves one file per weight beside the .onnx.
        with tempfile.TemporaryDirectory() as scratch:
            traced = Path(scratch) / "traced.onnx"
            export_static(module, example, input_name, output_name, traced)
            folded = MODELS / f"{clip.name}-{tower}.onnx"
            fold_shapes(traced, folded, traced_bytes(scratch))
        if clip.name in HALF_PRECISION:
            to_half_precision(folded)

    shutil.copy(clip.directory / "tokenizer.json", MODELS / f"{clip.name}.tokenizer.json")


def traced_bytes(scratch):
    """The size of the one tower just traced: its .onnx and any weights beside it."""
    return sum(file.stat().st_size for file in Path(scratch).iterdir())


def export_static(tower, example, input_name, output_name, path):
    torch.onnx.export(
        tower,
        (example,),
        # A string, not a Path: over 2 GB the exporter writes the weights beside it only for a string.
        str(path),
        input_names=[input_name],
        output_names=[output_name],
        opset_version=OPSET,
        dynamo=False,
    )


def fold_shapes(traced, folded, weights):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(folded)
    side_file = folded.with_name(folded.name + ".data")
    side_file.unlink(missing_ok=True)
    if weights > ONNX_FILE_LIMIT:
        options.add_session_config_entry("session.optimized_model_external_initializers_file_name", side_file.name)
        options.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "1024")
    ort.InferenceSession(str(traced), options, providers=["CPUExecutionProvider"])


def to_half_precision(path):
    """Rewrites a folded tower with half-precision weights. Its input and output
    stay single precision, with a Cast just inside, so the Rust side is the same
    for every size."""
    model = onnx.load(str(path))
    half = float16.convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=True, op_block_list=SINGLE_PRECISION_OPERATIONS
    )
    side_file = path.with_name(path.name + ".data")
    side_file.unlink(missing_ok=True)
    if sum(len(weight.raw_data) for weight in half.graph.initializer) > ONNX_FILE_LIMIT:
        onnx.save(half, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location=side_file.name)
    else:
        onnx.save(half, str(path))


def on_grey_canvas(crop, size):
    """The 224 crop in *.pixels.rgb, centred on a mid-grey square of the model's
    size. Both sides can build it exactly, so no larger crop has to be committed."""
    canvas = np.full((size, size, 3), 128, np.uint8)
    offset = (size - FIXTURE_CROP) // 2
    canvas[offset : offset + FIXTURE_CROP, offset : offset + FIXTURE_CROP] = crop
    return canvas


def vector_text(vector):
    return ",".join(f"{x:.8e}" for x in vector)


def write_references(clip):
    # pe-core-l14-336 -> pe-core-l
    stem = clip.name[: len("pe-core-l")]

    def write_table(kind, rows):
        (TEST_DATA / f"{stem}-{kind}-reference.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")

    ids = clip.token_ids(PROMPTS)
    rows = [f"# prompt\ttoken ids (open_clip, padded to {clip.context_length})\tunit vector (open_clip in PyTorch)"]
    for prompt, row_ids, vector in zip(PROMPTS, ids.tolist(), clip.embed_texts(PROMPTS)):
        rows.append(f"{prompt}\t{','.join(map(str, row_ids))}\t{vector_text(vector)}")
    write_table("text", rows)

    rows = [f"# fixture\tunit vector (open_clip in PyTorch on the crop in *.pixels.rgb, centred on a {clip.size} square of grey 128)"]
    for fixture in FIXTURES:
        crop = np.frombuffer((TEST_DATA / f"{fixture}.pixels.rgb").read_bytes(), np.uint8)
        canvas = on_grey_canvas(crop.reshape(FIXTURE_CROP, FIXTURE_CROP, 3), clip.size)
        rows.append(f"{fixture}\t{vector_text(clip.embed_pixels(normalise_to_chw(canvas)))}")
    write_table("image", rows)

    # The model's own config resizes with a bilinear filter and no crop. That is
    # a different vector: cosine 0.987 to 0.989 against this one for L on the fixtures.
    rows = ["# fixture\tunit vector (open_clip in PyTorch; bicubic resize and centre crop of the photo file, as preprocess.rs)"]
    for fixture in FIXTURES:
        square = FRAMINGS["centre-crop"](upright_rgb(MODELS / f"{fixture}.jpg"), clip.size)
        rows.append(f"{fixture}\t{vector_text(clip.embed_pixels(normalise_to_chw(square)))}")
    write_table("photo", rows)

    scale = clip.model.logit_scale
    print(f"{clip.name}: logit scale exp({scale.item():.4f}) = {scale.exp().item():.2f}")


def main(names):
    unknown = [name for name in names if name not in CHECKPOINTS]
    if unknown:
        sys.exit(f"unknown: {', '.join(unknown)}; known: {', '.join(CHECKPOINTS)}")
    for name in names:
        clip = PeCore(name)
        export(clip)
        write_references(clip)
        print(f"exported {name} to {MODELS}")


if __name__ == "__main__":
    main(sys.argv[1:] or [DEFAULT])
