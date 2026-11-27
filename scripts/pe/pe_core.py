"""Meta's PE-Core through open_clip in PyTorch, with this repo's framings redone in PIL.

This is the reference the Burn engine is gated against, for every size. The
checkpoints are the open_clip copies of Meta's, all Apache-2.0. G is the main
model; the others were distilled from it.
"""

import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from jina_clip import resize_and_centre_crop, unit_rows, upright_rgb

# name: (repository on Hugging Face, pinned revision)
CHECKPOINTS = {
    "pe-core-t16-384": ("timm/PE-Core-T-16-384", "7fe539ed578ac49a1c2b4f946e4b0747704c825a"),
    "pe-core-s16-384": ("timm/PE-Core-S-16-384", "3249a38eb1c432ec19231c6fe1774acb6a4e4efe"),
    "pe-core-b16-224": ("timm/PE-Core-B-16", "0038414f37721c5eafc1a5e0da802d291c909de3"),
    "pe-core-l14-336": ("timm/PE-Core-L-14-336", "8eff41b3f687e50a323662c2dda5eb3588c6dd35"),
    "pe-core-g14-448": ("timm/PE-Core-bigG-14-448", "17aa0c25addfa14198fa2ff73d845a22d433432e"),
}
FILES = [
    "open_clip_config.json",
    "open_clip_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]
# G's weights are 9.7 GB in single precision, more than an 8 GB card holds. It
# runs in half precision on the GPU here, and is exported to ONNX in half precision.
HALF_PRECISION = {"pe-core-g14-448"}

MEAN, STD = 0.5, 0.5
PAD_COLOUR = (128, 128, 128)


def resize_and_pad(image, size):
    """Mirrors preprocess.rs: the whole photo, centred on the mean colour."""
    width, height = image.size
    scaled_short = max(1, int(size * min(width, height) / max(width, height) + 0.5))
    new_width, new_height = (size, scaled_short) if width >= height else (scaled_short, size)
    square = Image.new("RGB", (size, size), PAD_COLOUR)
    square.paste(image.resize((new_width, new_height), Image.BICUBIC), ((size - new_width) // 2, (size - new_height) // 2))
    return square


def squash(image, size):
    """The model's own framing (open_clip_config.json): no crop, bilinear."""
    return image.resize((size, size), Image.BILINEAR)


# The first two are the framings `cli eval` gives every model, bicubic as in
# preprocess.rs, so a PyTorch score can be set beside a Burn one.
FRAMINGS = {"centre-crop": resize_and_centre_crop, "pad": resize_and_pad, "squash": squash}


def normalise_to_chw(square):
    scaled = np.asarray(square, np.float32) / 255.0
    return ((scaled - MEAN) / STD).transpose(2, 0, 1)


class PeCore:
    """Starts on the CPU in single precision. `to_gpu` moves it."""

    def __init__(self, name):
        repo, revision = CHECKPOINTS[name]
        self.name = name
        self.directory = Path(snapshot_download(repo, revision=revision, allow_patterns=FILES))
        self.model, _, self.own_transform = open_clip.create_model_and_transforms(f"local-dir:{self.directory}")
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(f"local-dir:{self.directory}")
        self.context_length = self.model.context_length
        size = self.model.visual.image_size
        self.size = size[0] if isinstance(size, (tuple, list)) else size
        self.device, self.half = "cpu", False

    def to_gpu(self):
        self.half = self.name in HALF_PRECISION
        self.model = (self.model.half() if self.half else self.model).cuda()
        self.device = "cuda"

    def token_ids(self, texts):
        return self.tokenizer(texts, context_length=self.context_length)

    @torch.no_grad()
    def embed_texts(self, texts):
        output = self.model.encode_text(self.token_ids(texts).to(self.device))
        return unit_rows(output.float().cpu().numpy())

    @torch.no_grad()
    def embed_pixels(self, chw):
        pixels = torch.from_numpy(chw[None].astype(np.float32)).to(self.device)
        output = self.model.encode_image(pixels.half() if self.half else pixels)
        return unit_rows(output.float().cpu().numpy())[0]

    def embed_image(self, path, framing="centre-crop"):
        return self.embed_pixels(normalise_to_chw(FRAMINGS[framing](upright_rgb(path), self.size)))
