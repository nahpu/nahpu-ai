"""Exports BioCLIP's two towers to ONNX and writes reference data for the Rust gates.

    scripts/bioclip/.venv/bin/python -I scripts/bioclip/export_onnx.py

Writes into models/:
    bioclip-text.onnx, bioclip-vision.onnx, bioclip.tokenizer.json
Writes into rust/crates/nahpu_embed_burn/tests/data/:
    bioclip-text-reference.tsv, bioclip-image-reference.tsv, bioclip-photo-reference.tsv

References come from open_clip in PyTorch, not from the exported ONNX, so the
gates check the export as well as Burn.

Shapes are fixed (batch 1) and folded by onnxruntime's basic optimiser. Without
that, nn.MultiheadAttention exports its reshapes as Shape/Mod/Slice arithmetic,
which burn-onnx 0.21 turns into code that does not compile.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import open_clip
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
TEST_DATA = ROOT / "rust/crates/nahpu_embed_burn/tests/data"
REPO = "imageomics/bioclip"
REVISION = "ce901ab3c6a913f9e9ef94ce6d27761069f4f01c"
CONTEXT_LENGTH = 77
OPSET = 17

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

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


def load():
    checkpoint = hf_hub_download(REPO, "open_clip_pytorch_model.bin", revision=REVISION)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained=checkpoint)
    model.eval()
    return model, preprocess


def export(model):
    with tempfile.TemporaryDirectory() as scratch:
        traced = Path(scratch) / "traced.onnx"

        export_static(TextTower(model), open_clip.tokenize(PROMPTS[:1]), "input_ids", "text_embeds", traced)
        fold_shapes(traced, MODELS / "bioclip-text.onnx")

        export_static(VisionTower(model), torch.zeros(1, 3, 224, 224), "pixel_values", "image_embeds", traced)
        fold_shapes(traced, MODELS / "bioclip-vision.onnx")

    tokenizer = hf_hub_download(REPO, "tokenizer.json", revision=REVISION)
    shutil.copy(tokenizer, MODELS / "bioclip.tokenizer.json")


def export_static(tower, example, input_name, output_name, path):
    torch.onnx.export(
        tower,
        (example,),
        path,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=OPSET,
        dynamo=False,
    )


def fold_shapes(traced, folded):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(folded)
    ort.InferenceSession(str(traced), options, providers=["CPUExecutionProvider"])


def normalise_to_chw(crop):
    return ((crop.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def unit(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def vector_text(vector):
    return ",".join(f"{x:.8e}" for x in vector)


def write_references(model, preprocess):
    with torch.no_grad():
        ids = open_clip.tokenize(PROMPTS)
        text = unit(model.encode_text(ids).numpy())
        rows = ["# prompt\ttoken ids (open_clip, padded to 77)\tunit vector (open_clip in PyTorch)"]
        for prompt, row_ids, vector in zip(PROMPTS, ids.tolist(), text):
            rows.append(f"{prompt}\t{','.join(map(str, row_ids))}\t{vector_text(vector)}")
        (TEST_DATA / "bioclip-text-reference.tsv").write_text("\n".join(rows) + "\n")

        rows = ["# fixture\tunit vector (open_clip in PyTorch on the PIL crop in *.pixels.rgb)"]
        for stem in FIXTURES:
            crop = np.frombuffer((TEST_DATA / f"{stem}.pixels.rgb").read_bytes(), np.uint8).reshape(224, 224, 3)
            pixels = torch.from_numpy(normalise_to_chw(crop)[None])
            rows.append(f"{stem}\t{vector_text(unit(model.encode_image(pixels).numpy())[0])}")
        (TEST_DATA / "bioclip-image-reference.tsv").write_text("\n".join(rows) + "\n")

        rows = ["# fixture\tunit vector (open_clip in PyTorch, its own preprocessing of the photo file)"]
        for stem in FIXTURES:
            pixels = preprocess(Image.open(MODELS / f"{stem}.jpg"))[None]
            rows.append(f"{stem}\t{vector_text(unit(model.encode_image(pixels).numpy())[0])}")
        (TEST_DATA / "bioclip-photo-reference.tsv").write_text("\n".join(rows) + "\n")

        print(f"logit scale: exp({model.logit_scale.item():.4f}) = {model.logit_scale.exp().item():.2f}")
        print(f"open_clip val transform: {preprocess}")


def main():
    model, preprocess = load()
    export(model)
    write_references(model, preprocess)
    print(f"exported to {MODELS}")


if __name__ == "__main__":
    main()
