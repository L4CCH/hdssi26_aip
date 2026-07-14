import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


EMBEDDINGS_NPY = "cluster_embeddings/global_embeddings.npy"
ASSIGNMENTS_JSON = (
    "cluster_embeddings/clusters/dbscan_assignments.json"
)
OUTPUT_PLOT = (
    "cluster_embeddings/clusters/dbscan_tsne_visualization.png"
)

EPS = 0.15
MIN_SAMPLES = 10
METRIC = "cosine"

PCA_COMPONENTS = 100
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42


print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_NPY)

print("Normalizing embeddings...")
embeddings = normalize(embeddings)

print("Loading cluster assignments...")
with open(ASSIGNMENTS_JSON, "r", encoding="utf-8") as file:
    assignments = json.load(file)

cluster_labels = np.array(
    [item["cluster"] for item in assignments]
)

if len(embeddings) != len(cluster_labels):
    raise ValueError(
        "The number of embeddings does not match the number of "
        "cluster assignments.\n"
        f"Embeddings: {len(embeddings)}\n"
        f"Assignments: {len(cluster_labels)}"
    )

print("Removing DBSCAN noise points (-1)...")

mask = cluster_labels != -1

filtered_embeddings = embeddings[mask]
filtered_labels = cluster_labels[mask]

print(f"Remaining images: {len(filtered_embeddings)}")
print(f"Clusters shown: {len(np.unique(filtered_labels))}")

if len(filtered_embeddings) == 0:
    raise ValueError(
        "No clustered images remain after removing DBSCAN noise points."
    )

# PCA
effective_pca_components = min(
    PCA_COMPONENTS,
    filtered_embeddings.shape[0],
    filtered_embeddings.shape[1],
)

print(
    f"Running PCA to {effective_pca_components} dimensions..."
)

pca = PCA(
    n_components=effective_pca_components,
    random_state=TSNE_RANDOM_STATE,
)

pca_embeddings = pca.fit_transform(filtered_embeddings)

explained_variance = (
    np.sum(pca.explained_variance_ratio_) * 100
)

print(
    "Explained variance:",
    round(explained_variance, 2),
    "%",
)

# t-SNE
effective_perplexity = min(
    TSNE_PERPLEXITY,
    len(pca_embeddings) - 1,
)

if effective_perplexity < 1:
    raise ValueError(
        "At least two clustered images are required to run t-SNE."
    )

print(
    f"Running t-SNE with perplexity={effective_perplexity}..."
)

tsne = TSNE(
    n_components=2,
    perplexity=effective_perplexity,
    init="pca",
    random_state=TSNE_RANDOM_STATE,
    learning_rate="auto",
)

tsne_embeddings = tsne.fit_transform(pca_embeddings)

print("t-SNE complete.")

# Graphing
plt.figure(figsize=(14, 10))

scatter = plt.scatter(
    tsne_embeddings[:, 0],
    tsne_embeddings[:, 1],
    c=filtered_labels,
    cmap="tab20",
    s=10,
    alpha=0.8,
)

plt.title(
    "DBSCAN Cluster Visualization\n"
    f"eps={EPS}, "
    f"min_samples={MIN_SAMPLES}, "
    f"metric='{METRIC}'\n"
    f"PCA({effective_pca_components}) → t-SNE\n"
    "Noise points removed"
)

plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.colorbar(
    scatter,
    label="Cluster ID",
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight",
)

print("Saved visualization to:")
print(OUTPUT_PLOT)

# Terminal error checking to compare the number of embeddings
# with the cluster summary JSON.
print("Unique cluster labels:")
print(np.unique(filtered_labels))

plt.show()
