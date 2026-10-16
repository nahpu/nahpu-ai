"""Prints the onnxruntime zero-shot ranking for photos, to compare with `cli classify`.

    scripts/reference/.venv/bin/python -I scripts/reference/classify.py LABELS PROMPTS WORDING PHOTO...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jina_clip import JinaClip
from zero_shot import class_vectors, read_labels


def main(labels_path, prompts, wording, *photos):
    clip = JinaClip()
    labels = read_labels(labels_path)
    classes = class_vectors(clip, labels, wording, prompts)
    for photo in photos:
        cosines = classes @ clip.embed_image(photo)
        print(photo)
        for i in cosines.argsort()[::-1][:5]:
            print(f"  {labels[i]['class']:<16} cosine {cosines[i]:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
