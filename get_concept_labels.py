"""
get_concept_labels.py

Runs zero-shot concept labeling using CLIP embeddings.

Inputs:
- embeddings/global_embeddings.npy
- embeddings/global_embeddings.txt

Output:
- embeddings/zero_shot_labels_top3.json
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os


IMAGE_EMBEDDINGS_NPY = "embeddings/global_embeddings.npy"
IMAGE_EMBEDDINGS_TXT = "embeddings/global_embeddings.txt"

OUTPUT_FILE = "embeddings/zero_shot_labels_top3.json"

TOP_K = 3

LABELS = [
    "man",
    "woman",
    "group of men",
    "group of women",
    "machine"
]

# Assign "Other" when the best label score is below this threshold.
# You will likely need to inspect results and adjust this value.
OTHER_LABEL = "Other"
OTHER_THRESHOLD = 0.24


print("Loading model: sentence-transformers/clip-ViT-B-32")
clip_model = SentenceTransformer("clip-ViT-B-32")

print("Text embedding dim:", clip_model.get_sentence_embedding_dimension())

print("Encoding text labels...")
text_vecs = clip_model.encode(
    LABELS,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)

print("text_vecs shape:", text_vecs.shape)

print("Loading image embeddings...")
image_vecs = np.load(IMAGE_EMBEDDINGS_NPY).astype(np.float32)

print("image_vecs shape:", image_vecs.shape)

print("Normalizing image embeddings...")
norms = np.linalg.norm(image_vecs, axis=1, keepdims=True)

# Avoid division by zero for any empty/invalid embedding.
norms[norms == 0] = 1.0

image_vecs = image_vecs / norms

print("Loading filenames...")
with open(IMAGE_EMBEDDINGS_TXT, "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f if line.strip()]

if len(filenames) != image_vecs.shape[0]:
    raise ValueError(
        f"Number of filenames ({len(filenames)}) does not match "
        f"number of embeddings ({image_vecs.shape[0]})"
    )

if TOP_K > len(LABELS):
    raise ValueError("TOP_K cannot be larger than the number of labels.")

print("Computing image-text similarities...")
scores = image_vecs @ text_vecs.T

results = []

print(f"Selecting top {TOP_K} labels per image...")
for i, filename in enumerate(filenames):
    ordered = np.argsort(scores[i])[::-1][:TOP_K]

    top_labels = [
        {
            "label": LABELS[j],
            "score": float(scores[i, j])
        }
        for j in ordered
    ]

    raw_top1_label = top_labels[0]["label"]
    raw_top1_score = top_labels[0]["score"]

    # Apply Other-category logic.
    if raw_top1_score >= OTHER_THRESHOLD:
        assigned_label = raw_top1_label
    else:
        assigned_label = OTHER_LABEL

    results.append({
        "filename": filename,

        # Final label after applying the threshold.
        "assigned_label": assigned_label,

        # Raw top prediction before applying the threshold.
        "raw_top1_label": raw_top1_label,
        "raw_top1_score": raw_top1_score,

        "other_threshold": OTHER_THRESHOLD,

        # Preserve the original top-k results for inspection.
        f"top{TOP_K}": top_labels
    })

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Saved zero-shot labels to {OUTPUT_FILE}")

other_count = sum(
    result["assigned_label"] == OTHER_LABEL
    for result in results
)

print(
    f"Assigned {other_count}/{len(results)} images "
    f"({other_count / max(len(results), 1) * 100:.2f}%) to '{OTHER_LABEL}'."
)
