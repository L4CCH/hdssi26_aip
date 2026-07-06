"""
get_clusters.py

Runs DBSCAN clustering on image embeddings and maps each embedding row
back to its corresponding filename from global_embeddings.txt.

Inputs:
- embeddings/global_embeddings.npy
- embeddings/global_embeddings.txt

Outputs:
- embeddings/clusters/dbscan_assignments.json
- embeddings/clusters/clusters.json
"""

import os
import json
import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize


EMBEDDINGS_NPY = "embeddings/global_embeddings.npy"
EMBEDDINGS_TXT = "embeddings/global_embeddings.txt"

OUTPUT_DIR = "embeddings/clusters"
OUTPUT_ASSIGNMENTS = os.path.join(OUTPUT_DIR, "dbscan_assignments.json")
OUTPUT_CLUSTERS = os.path.join(OUTPUT_DIR, "clusters.json")
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR, "cluster_summary.json")

# Feel free to change these!!
EPS = 0.2
MIN_SAMPLES = 2
N_JOBS = -1
METRIC = "cosine"


print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_NPY)

# normalize embeddings before cosine-based DBSCAN
embeddings = normalize(embeddings)

print("Loading filename mapping...")
with open(EMBEDDINGS_TXT, "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f if line.strip()]

if len(embeddings) != len(filenames):
    raise ValueError(
        f"Mismatch: {len(embeddings)} embeddings but {len(filenames)} filenames. "
        "The .npy rows must match the .txt lines exactly."
    )

print(f"Loaded {len(embeddings)} embeddings and {len(filenames)} filenames.")


print("Running DBSCAN...")
db = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES,
    metric=METRIC,
    n_jobs=N_JOBS
).fit(embeddings)

labels = db.labels_

print("DBSCAN complete.")


# row-level assignments
assignments = []

for i, (filename, cluster_id) in enumerate(zip(filenames, labels)):
    assignments.append({
        "embedding_index": i,
        "filename": filename,
        "cluster": int(cluster_id)
    })


# cluster dictionary
clusters = defaultdict(list)

for filename, cluster_id in zip(filenames, labels):
    clusters[str(int(cluster_id))].append(filename)

clusters = dict(clusters)


cluster_counts = Counter(labels)

summary = {
    "total_embeddings": int(len(embeddings)),
    "eps": EPS,
    "min_samples": MIN_SAMPLES,
    "metric": METRIC,
    "num_clusters_excluding_noise": int(len([c for c in cluster_counts if c != -1])),
    "num_noise_points": int(cluster_counts.get(-1, 0)),
    "cluster_sizes": {
        str(int(cluster_id)): int(count)
        for cluster_id, count in sorted(cluster_counts.items())
    }
}


os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_ASSIGNMENTS, "w", encoding="utf-8") as f:
    json.dump(assignments, f, indent=2)

with open(OUTPUT_CLUSTERS, "w", encoding="utf-8") as f:
    json.dump(clusters, f, indent=2)

with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Saved row-level assignments to: {OUTPUT_ASSIGNMENTS}")
print(f"Saved cluster dictionary to: {OUTPUT_CLUSTERS}")
print(f"Saved summary to: {OUTPUT_SUMMARY}")

print("\nSummary:")
print(json.dumps(summary, indent=2))