"""
run_tsne.py

Runs t-SNE on CLIP image embeddings and labels each point with
its corresponding filename.

Inputs:
- embeddings/global_embeddings.npy
- embeddings/global_embeddings.txt

The filename on line N of global_embeddings.txt must correspond to
row N of global_embeddings.npy.

Outputs:
- embeddings/tsne_2d.npy
- embeddings/tsne_plot_labeled.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Input files
EMBEDDINGS_NPY = "embeddings/global_embeddings.npy"
FILENAMES_TXT = "embeddings/global_embeddings.txt"

# Output files
OUTPUT_COORDINATES = "embeddings/tsne_2d.npy"
OUTPUT_PLOT = "embeddings/tsne_plot_labeled.png"

# t-SNE settings
RANDOM_STATE = 42

"""
The maximum number of dimensions to keep after PCA before running t-SNE. 
PCA reduces the original 512-dimensional CLIP embeddings to at most 50 dimensions, 
which speeds up t-SNE while preserving most of the important structure. 
If there are too few samples (like your 42 images), the script automatically 
skips or reduces PCA.
"""
MAX_PCA_COMPONENTS = 50

"""
The target perplexity for t-SNE, which controls the effective number of neighboring 
points each point considers when constructing the low-dimensional embedding. 
Larger values emphasize broader/global structure, while smaller values focus more 
on local clusters. The script automatically lowers this value if the dataset is too small.
"""
DEFAULT_PERPLEXITY = 30

# Plot settings
FIGURE_WIDTH = 14
FIGURE_HEIGHT = 10
POINT_SIZE = 35
LABEL_FONT_SIZE = 7
PLOT_DPI = 300

# When True, labels include the full path.
# When False, labels show only the final filename.
SHOW_FULL_PATH = False


print("Loading embeddings...")

embeddings = np.load(
    EMBEDDINGS_NPY
).astype(np.float32)

print("Embedding shape:", embeddings.shape)


# Validate embedding array
if embeddings.ndim != 2:
    raise ValueError(
        "Expected embeddings to be a 2D array, "
        f"but got shape {embeddings.shape}."
    )

n_samples, n_features = embeddings.shape

if n_samples < 2:
    raise ValueError(
        "At least 2 embeddings are required to run t-SNE."
    )

if not np.all(np.isfinite(embeddings)):
    raise ValueError(
        "Embeddings contain NaN or infinite values."
    )


print("Loading filenames...")

with open(
    FILENAMES_TXT,
    "r",
    encoding="utf-8"
) as file:
    filenames = [
        line.strip()
        for line in file
        if line.strip()
    ]

print("Number of filenames:", len(filenames))


# Confirm that each embedding has one corresponding filename
if len(filenames) != n_samples:
    raise ValueError(
        "The number of filenames does not match the number "
        "of embedding rows.\n"
        f"Embedding rows: {n_samples}\n"
        f"Filenames: {len(filenames)}"
    )


# Use either the full path or only the base filename as the label
if SHOW_FULL_PATH:
    plot_labels = filenames
else:
    plot_labels = [
        os.path.basename(filename)
        for filename in filenames
    ]


# Normalize embeddings if necessary
norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

if np.any(norms == 0):
    raise ValueError(
        "One or more embeddings have a norm of zero."
    )

if not np.allclose(
    norms,
    1.0,
    atol=1e-2
):
    print("Normalizing embeddings...")
    embeddings = embeddings / norms
else:
    print("Embeddings are already normalized.")


# Automatically choose a valid perplexity
perplexity = min(
    DEFAULT_PERPLEXITY,
    max(2, (n_samples - 1) // 3)
)

# t-SNE requires perplexity to be smaller than n_samples
perplexity = min(
    perplexity,
    n_samples - 1
)

print("Using perplexity:", perplexity)


# Use PCA before t-SNE when there are enough samples
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

print(
    "t-SNE coordinate shape:",
    embeddings_2d.shape
)


# Create output directory if needed
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


# Create labeled plot
print("Creating labeled visualization...")

plt.figure(
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT
    )
)

plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    s=POINT_SIZE,
    alpha=0.75
)


# Add one filename label to each point
for index, label in enumerate(plot_labels):

    x_coordinate = embeddings_2d[index, 0]
    y_coordinate = embeddings_2d[index, 1]

    plt.annotate(
        label,
        xy=(
            x_coordinate,
            y_coordinate
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=LABEL_FONT_SIZE,
        alpha=0.85
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
    dpi=PLOT_DPI,
    bbox_inches="tight"
)

print(
    "Saved labeled visualization to:",
    OUTPUT_PLOT
)

plt.show()

print("Done!")