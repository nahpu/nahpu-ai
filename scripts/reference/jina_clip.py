"""jina-clip-v1 through onnxruntime, with the torchvision transform redone in PIL.

This is the reference the Rust engines are gated against. It runs the same ONNX
files, so any difference comes from our preprocessing or from Burn.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
MAX_TOKENS = 512


def upright_rgb(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def resize_and_centre_crop(image, size=SIZE):
    width, height = image.size
    if width <= height:
        new_width, new_height = size, int(size * height / width)
    else:
        new_width, new_height = int(size * width / height), size
    resized = image.resize((new_width, new_height), Image.BICUBIC)
    top = int(round((new_height - size) / 2.0))
    left = int(round((new_width - size) / 2.0))
    return resized.crop((left, top, left + size, top + size))


def normalise_to_chw(square):
    scaled = np.asarray(square, np.float32) / 255.0
    return ((scaled - MEAN) / STD).transpose(2, 0, 1)


def unit_rows(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


class JinaClip:
    def __init__(self, models=MODELS):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self.text = ort.InferenceSession(str(models / "jina-clip-v1-text.onnx"), options, providers=providers)
        self.vision = ort.InferenceSession(str(models / "jina-clip-v1-vision.onnx"), options, providers=providers)
        self.tokenizer = Tokenizer.from_file(str(models / "jina-clip-v1.tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0)

    def token_ids(self, texts):
        return np.array([e.ids for e in self.tokenizer.encode_batch(texts)], np.int64)

    def embed_texts(self, texts):
        return unit_rows(self.text.run(None, {"input_ids": self.token_ids(texts)})[0])

    def embed_pixels(self, chw):
        return unit_rows(self.vision.run(None, {"pixel_values": chw[None].astype(np.float32)})[0])[0]

    def embed_image(self, path):
        return self.embed_pixels(normalise_to_chw(resize_and_centre_crop(upright_rgb(path))))
