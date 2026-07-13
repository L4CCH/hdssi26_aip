"""
run_tsne.py

Runs t-SNE on CLIP image embeddings and creates a visualization
colored by existing DBSCAN cluster assignments.

Inputs:
- embeddings/global_embeddings.npy
- embeddings/clusters/dbscan_assignments.json

Outputs:
- embeddings/tsne_2d.npy
- embeddings/tsne_plot_by_cluster.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


INPUT_FILE = "embeddings/global_embeddings.npy"
CLUSTER_ASSIGNMENTS_FILE = (
    "embeddings/clusters/dbscan_assignments.json"
)

OUTPUT_COORDINATES = "embeddings/tsne_2d.npy"
OUTPUT_PLOT = "embeddings/tsne_plot_by_cluster.png"

RANDOM_STATE = 42
MAX_PCA_COMPONENTS = 50
DEFAULT_PERPLEXITY = 30

POINT_SIZE = 30
POINT_ALPHA = 0.8
NOISE_ALPHA = 0.4


print("Loading embeddings...")

embeddings = np.load(
    INPUT_FILE
).astype(np.float32)

print(
    "Embedding shape:",
    embeddings.shape
)


# Validate embedding array
if embeddings.ndim != 2:
    raise ValueError(
        f"Expected a 2D embedding array, "
        f"but got shape {embeddings.shape}"
    )

n_samples, n_features = embeddings.shape

if n_samples < 2:
    raise ValueError(
        "At least 2 embeddings are required "
        "to run t-SNE."
    )


# Check for invalid values
if not np.all(
    np.isfinite(embeddings)
):
    raise ValueError(
        "Embeddings contain NaN or infinite values."
    )


# Load DBSCAN cluster assignments
print("Loading cluster assignments...")

with open(
    CLUSTER_ASSIGNMENTS_FILE,
    "r",
    encoding="utf-8"
) as file:
    assignments = json.load(file)


# Support several common JSON formats
if isinstance(assignments, list):

    cluster_labels = []

    for index, item in enumerate(assignments):

        if not isinstance(item, dict):
            raise ValueError(
                "Each item in the cluster assignment "
                "list must be a dictionary."
            )

        if "cluster" in item:
            cluster_labels.append(
                item["cluster"]
            )

        elif "cluster_id" in item:
            cluster_labels.append(
                item["cluster_id"]
            )

        elif "label" in item:
            cluster_labels.append(
                item["label"]
            )

        else:
            raise KeyError(
                f"Could not find a cluster label "
                f"for assignment at index {index}. "
                f"Expected 'cluster', 'cluster_id', "
                f"or 'label'."
            )


elif isinstance(assignments, dict):

    if "assignments" in assignments:

        assignment_list = assignments[
            "assignments"
        ]

        cluster_labels = []

        for index, item in enumerate(
            assignment_list
        ):

            if "cluster" in item:
                cluster_labels.append(
                    item["cluster"]
                )

            elif "cluster_id" in item:
                cluster_labels.append(
                    item["cluster_id"]
                )

            elif "label" in item:
                cluster_labels.append(
                    item["label"]
                )

            else:
                raise KeyError(
                    f"Could not find a cluster label "
                    f"for assignment at index {index}."
                )

    else:
        # Supports a dictionary such as:
        # {
        #     "image1.jpg": 0,
        #     "image2.jpg": -1
        # }
        cluster_labels = list(
            assignments.values()
        )

else:
    raise ValueError(
        "Unsupported cluster assignment JSON format."
    )


cluster_labels = np.asarray(
    cluster_labels,
    dtype=int
)

print(
    "Cluster label shape:",
    cluster_labels.shape
)


# Confirm that every embedding has one cluster label
if len(cluster_labels) != n_samples:
    raise ValueError(
        f"Number of embeddings ({n_samples}) does not "
        f"match number of cluster labels "
        f"({len(cluster_labels)}). "
        f"The assignment order must match the embedding "
        f"row order."
    )


unique_clusters = np.unique(
    cluster_labels
)

non_noise_clusters = unique_clusters[
    unique_clusters != -1
]

noise_count = int(
    np.sum(cluster_labels == -1)
)

print(
    "Number of non-noise clusters:",
    len(non_noise_clusters)
)

print(
    "Number of noise points:",
    noise_count
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

if not np.allclose(
    norms,
    1.0,
    atol=1e-2
):
    print("Normalizing embeddings...")

    embeddings = embeddings / norms

else:
    print(
        "Embeddings are already normalized."
    )


# Automatically choose a valid perplexity
perplexity = min(
    DEFAULT_PERPLEXITY,
    max(
        2,
        (n_samples - 1) // 3
    )
)

perplexity = min(
    perplexity,
    n_samples - 1
)

print(
    "Using perplexity:",
    perplexity
)


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


# Create output directories
coordinate_output_directory = os.path.dirname(
    OUTPUT_COORDINATES
)

plot_output_directory = os.path.dirname(
    OUTPUT_PLOT
)

if coordinate_output_directory:
    os.makedirs(
        coordinate_output_directory,
        exist_ok=True
    )

if plot_output_directory:
    os.makedirs(
        plot_output_directory,
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


# Create cluster-colored visualization
print(
    "Creating cluster-colored visualization..."
)

plt.figure(
    figsize=(12, 9)
)


# Use a continuous colormap so the script can
# support more than 20 clusters
colormap = plt.get_cmap(
    "nipy_spectral"
)

number_of_clusters = len(
    non_noise_clusters
)


# Plot each non-noise cluster separately
for color_index, cluster_id in enumerate(
    non_noise_clusters
):

    cluster_mask = (
        cluster_labels == cluster_id
    )

    if number_of_clusters <= 1:
        color_position = 0.5
    else:
        color_position = (
            color_index
            / number_of_clusters
        )

    cluster_color = colormap(
        color_position
    )

    plt.scatter(
        embeddings_2d[
            cluster_mask,
            0
        ],
        embeddings_2d[
            cluster_mask,
            1
        ],
        color=cluster_color,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        label=f"Cluster {cluster_id}"
    )


# Plot DBSCAN noise separately in gray
noise_mask = (
    cluster_labels == -1
)

if np.any(noise_mask):
    plt.scatter(
        embeddings_2d[
            noise_mask,
            0
        ],
        embeddings_2d[
            noise_mask,
            1
        ],
        color="lightgray",
        edgecolors="none",
        s=POINT_SIZE,
        alpha=NOISE_ALPHA,
        label="Noise"
    )


plt.title(
    "t-SNE Visualization of CLIP Embeddings "
    "Colored by DBSCAN Cluster"
)

plt.xlabel(
    "t-SNE Dimension 1"
)

plt.ylabel(
    "t-SNE Dimension 2"
)


# Only show a legend when there are not too many
# clusters. Large legends can cover the entire plot.
if number_of_clusters <= 20:

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        markerscale=1.2
    )

else:
    print(
        "Skipping plot legend because there are "
        f"{number_of_clusters} clusters."
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight"
)

print(
    "Saved cluster-colored t-SNE visualization to:",
    OUTPUT_PLOT
)

plt.show()


print("Done!")

print(
    "Output coordinate shape:",
    embeddings_2d.shape
)