"""
run_tsne.py

Runs t-SNE on CLIP image embeddings and creates a visualization.

Input:
- embeddings/global_embeddings.npy

Outputs:
- embeddings/tsne_2d.npy
- embeddings/tsne_plot.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


INPUT_FILE = "embeddings/global_embeddings.npy"

OUTPUT_COORDINATES = "embeddings/tsne_2d.npy"
OUTPUT_PLOT = "embeddings/tsne_plot.png"

RANDOM_STATE = 42
MAX_PCA_COMPONENTS = 50
DEFAULT_PERPLEXITY = 30


print("Loading embeddings...")

embeddings = np.load(INPUT_FILE).astype(np.float32)

print("Embedding shape:", embeddings.shape)


# Validate embedding array
if embeddings.ndim != 2:
    raise ValueError(
        f"Expected a 2D embedding array, but got shape {embeddings.shape}"
    )

n_samples, n_features = embeddings.shape

if n_samples < 2:
    raise ValueError(
        "At least 2 embeddings are required to run t-SNE."
    )


# Check for invalid values
if not np.all(np.isfinite(embeddings)):
    raise ValueError(
        "Embeddings contain NaN or infinite values."
    )


# Normalize embeddings if needed
norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

if np.any(norms == 0):
    raise ValueError(
        "One or more embeddings have a norm of 0."
    )

if not np.allclose(norms, 1.0, atol=1e-2):
    print("Normalizing embeddings...")

    embeddings = embeddings / norms
else:
    print("Embeddings are already normalized.")


# Automatically choose a valid perplexity
perplexity = min(
    DEFAULT_PERPLEXITY,
    max(2, (n_samples - 1) // 3)
)

perplexity = min(
    perplexity,
    n_samples - 1
)

print("Using perplexity:", perplexity)


# PCA is useful for larger datasets
if n_samples > MAX_PCA_COMPONENTS:

    pca_components = min(
        MAX_PCA_COMPONENTS,
        n_samples - 1,
        n_features
    )

    print(
        f"Running PCA "
        f"({n_features} -> {pca_components})..."
    )

    pca = PCA(
        n_components=pca_components,
        random_state=RANDOM_STATE
    )

    tsne_input = pca.fit_transform(
        embeddings
    )

    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )

    print(
        "PCA explained variance:",
        f"{explained_variance:.4f}"
    )

else:
    print(
        f"Skipping PCA because there are only "
        f"{n_samples} embeddings."
    )

    tsne_input = embeddings


# Run t-SNE
print("Running t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    learning_rate="auto",
    init="pca",
    random_state=RANDOM_STATE
)

embeddings_2d = tsne.fit_transform(
    tsne_input
)


# Create output directory
output_directory = os.path.dirname(
    OUTPUT_COORDINATES
)

if output_directory:
    os.makedirs(
        output_directory,
        exist_ok=True
    )


# Save t-SNE coordinates
np.save(
    OUTPUT_COORDINATES,
    embeddings_2d
)

print(
    "Saved t-SNE coordinates to:",
    OUTPUT_COORDINATES
)


# Create visualization
print("Creating visualization...")

plt.figure(
    figsize=(10, 8)
)

plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    s=30,
    alpha=0.75
)

plt.title(
    "t-SNE Visualization of CLIP Embeddings"
)

plt.xlabel(
    "t-SNE Dimension 1"
)

plt.ylabel(
    "t-SNE Dimension 2"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight"
)

print(
    "Saved t-SNE visualization to:",
    OUTPUT_PLOT
)

plt.show()


print("Done!")
print(
    "Output coordinate shape:",
    embeddings_2d.shape
)