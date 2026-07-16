# Methodologies

## Table of Contents

- [GLiNER2 Relation Extraction](#gliner2-relation-extraction)
- [DBSCAN Clustering](#dbscan-clustering)
- [Zero-Shot Concept Labeling](#zero-shot-concept-labeling)

## GLiNER2 Relation Extraction

### Overview

The relation extraction pipeline is implemented in [`relation_extraction.ipynb`](relation_extraction.ipynb). The notebook identifies semantic relationships between entities mentioned throughout the oral history collection. Whereas Named Entity Recognition (NER) identifies individual entities (people, organizations, locations, etc.), relation extraction builds upon these entities, identifying **how they are connected**. We then construct knowledge graphs to visualization these relationships.

One of GLiNER2's distinguishing features is that it performs open-schema relation extraction. Instead of being limited to a fixed set of predefined relation labels, we are able to provide the relation types we want the model to look for at inference time. Our custom set of labels is as follows:

* `collaborated_with`
* `supervised_by`
* `worked_at`
* `worked_on`
* `supported_by`
* `assisted_by`
* `married_to`
* `parent_of`
* `child_of`
* `sibling_of`

The resulting structured data can be used to analyze collaborations, institutional networks, scientific careers, and underrecognized contributors across the archive.

For each relation type, we supply a `description` explaining what the relation means. GLiNER2 uses this text to understand the semantics of the relation label. For example, instead of only seeing the label `sibling_of`, the model also sees: `Family relationship between two named siblings, including brothers and sisters.` 

Additionally, the `threshold` is the minimum confidence score required for GLiNER2 to keep a predicted relation.

---

The `relation_extraction.ipynb` notebook is designed to run in **Google Colab** using a **GPU** runtime.

Using an A100 GPU, relation extraction across all **1,754 unique oral history transcripts** completes in approximately **2 hours**.

Model Used: `fastino/gliner2-base-v1`

The notebook performs the following steps:

1. Loads the cleaned transcript dataset (`bib_interviewee_date_body.json`).
2. Splits transcript text into overlapping chunks due to model context window limitations and relation extraction quality.
3. Runs GLiNER2 relation extraction on every chunk.
4. Consolidates duplicate relation triples extracted from multiple chunks. A relation triple consists of `(head entity, relation, tail entity)`, for example: `(Richard Feyman, worked_at, Los Alamos)`.
5. Aggregates corpus-wide statistics.
6. Generates visualizations summarizing extracted relationships.
7. Builds an interactive knowledge graph of highly connected entities.

---

The primary output is:

```
extracted_relations/
└── gliner2_relations_consolidated.json
```

The dataset contains **72,692 unique extracted relations**.

Each JSON object represents one unique relation triple aggregated across the entire corpus.

Each record contains the following fields.

| Field                  | Description                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `relation`             | Type of relationship (e.g. `"worked_at"`).                                                                  |
| `head`                 | Source entity.                                                                                              |
| `tail`                 | Target entity.                                                                                              |
| `transcript_count`     | Number of unique transcripts in which the relation appears.                                                 |
| `total_count`          | Total number of occurrences across all transcripts, including repeated mentions within the same transcript. |
| `counts_by_transcript` | Dictionary mapping transcript bib numbers to occurrence counts.                                             |
| `examples`             | List containing at most one representative extraction example.                                              |

The `counts_by_transcript` dictionary has the structure:

```
{
    "48389": 2,
    "47291": 1,
    ...
}
```

where:

* key = transcript bib number
* value = number of occurrences within that transcript

Each example contains:

| Field              | Description                                              |
| ------------------ | -------------------------------------------------------- |
| `field_bib_number` | Transcript bib number.                                   |
| `interviewee`      | Interviewee name.                                        |
| `interview_date`   | Interview date.                                          |
| `chunk_index`      | Chunk from which the relation was extracted.             |
| `head_confidence`  | Confidence score for the head entity (currently set to `null`). |
| `tail_confidence`  | Confidence score for the tail entity (currently set to `null`). |
| `context`          | Text snippet from which the relation was extracted.      |


Although the parsing code supports relation outputs containing entity confidence scores, the GLiNER2 model used in this project returned relations as `(head, tail)` pairs without confidence metadata. Our analysis focused on the extracted relationships rather than entity-levevl confidence estimates. Consequently, the `head_confidence` and `tail_confidence` fields are stored as `null` in the output JSON.

---

The notebook computes corpus-wide statistics, including:

* Distribution of extracted relation types
* Most frequent relation triples
* Relations ranked by total occurrence count
* Relations ranked by number of transcripts in which they appear

These summaries help identify the most common relationship patterns throughout the archive.

---

To visualize our extracted relations, we opted for:

### Knowledge Graphs

Each provides a high-level view of important people, organizations, laboratories, projects, and other entities while illustrating how they are connected.

#### Nodes

Each node represents an extracted entity. Node size reflects the selected ranking metric. Each node is annotated with its ranking score. Two ranking metrics are available: **frequency** and **connectivity**. 

Frequency ranks entities by the total number of extracted relation occurrences involving that entity. This highlights entities that appear most frequently throughout the corpus. Connectivity ranks entities by the number of unique entities directly connected to them. This highlights entities that serve as hubs within the relationship network.

#### Edges

Each directed edge represents one extracted relationship between two entities. Edge colors correspond to different relation types. Edge thickness is proportional to the total number of times that specific triple was extracted across the corpus. 

#### Entity Selection

To improve readability, only the **Top X entities** are displayed.

Entities may be selected using either frequency or connectivity.

#### Graph Filtering

Two visualization modes are available.

1. Require Both Entities in Top X

Only relationships where **both entities** belong to the selected Top X are displayed.

This produces a compact graph showing the strongest relationships among the highest-ranked entities.

2. Require At Least One Entity in Top X

Relationships are displayed whenever **either endpoint** belongs to the Top X.

This reveals how highly ranked entities connect to the broader network while preserving additional context. **Note: when this option is selected, graphs take substantially longer time to generate.**

---

We generated the following knowledge graph visualizations for the 20 entities ranked highest by frequency and connectivity:

### Example Outputs

![Top 20 by connectivity](readme_images/top20_connections.png)

![Top 20 by frequency](readme_images/top20_frequency.png)

---

## DBSCAN Clustering
### Overview

The clustering pipeline applies Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to identify groups of visually similar images within the Emilio Segrè Visual Archives image collection. Unlike traditional supervised classification approaches, DBSCAN does not require predefined identity labels. Instead, it groups images based on similarity within the embedding space, allowing clusters of potentially related individuals to emerge automatically.

Before clustering, each image is converted into a numerical representation called an embedding. These embeddings encode visual characteristics of each image into a high-dimensional vector space, where images with similar facial features are expected to have smaller distances between their corresponding vectors. DBSCAN operates on these embeddings to identify dense regions representing potential groups of related images while separating ambiguous images as noise.

The goal of this pipeline is not to assign definitive identities to individuals, but to explore computational methods discovering meaningful patterns of similarity across a large-scale historical image collection.

### Model and Clustering Method

The clustering pipeline uses DBSCAN because it provides two important advantages for archival image analysis:

- It does not require specifying the expected number of clusters beforehand, which is useful when the number of individuals represented in a historical image collection is unknown.
- It can identify images that do not clearly belong to any group by assigning them as noise points rather than forcing every image into a cluster.

DBSCAN relies on two primary clustering parameters:

| Parameter | Description | Reasoning |
| --- | --- | --- |
| `eps` | Maximum distance between two embeddings for them to be considered neighbors. | Controls how similar two images must be in embedding space to be grouped together. Smaller values create stricter clusters, while larger values allow more images to be grouped together. |
| `min_samples` | Minimum number of neighboring images required for a dense region to form a cluster. | Controls the minimum evidence required before identifying a group. Higher values produce more conservative clusters and classify more images as noise. |

The initial implementation used manually selected values for `eps` and `min_samples`. Multiple parameter combinations were tested, and the resulting clusters were evaluated through visualization and qualitative inspection of whether images with similar facial characteristics appeared together.

### Visualization Parameters

After clustering, dimensionality reduction was applied to visualize the high-dimensional image embeddings. The following parameters were selected for PCA and t-SNE visualization:

| Parameter | Description | Reasoning |
| --- | --- | --- |
| `PCA_COMPONENTS = 100` | Number of dimensions retained after Principal Component Analysis. | Reduces the high-dimensional embedding space while preserving important variance before applying t-SNE. |
| `TSNE_RANDOM_STATE = 42` | Random seed used for reproducible t-SNE results. | Ensures that repeated runs produce comparable visualizations. |

These visualization parameters were selected to balance computational efficiency and preservation of meaningful relationships between image embeddings.

### Dimensionality Reduction and Visualization

Image embeddings exist in a high-dimensional feature space, where each image is represented by a numerical vector containing many dimensions. While these embeddings are useful for measuring similarity between images, they cannot be directly visualized or easily interpreted. Dimensionality reduction techniques were therefore applied to project the embeddings into a lower-dimensional space while preserving meaningful relationships between images.

The visualization pipeline follows:

```
Image and File Embeddings
        ↓
DBSCAN Clustering
        ↓
DBSCAN Assignments (cluster JSON)
        ↓
Remove Noise Points (-1) (implemented in run_tsne.py)
        ↓
Principal Component Analysis (PCA) (implemented in run_tsne.py)
        ↓
t-SNE Visualization
        ↓
dbscan_tsne_visualization.png
```

After DBSCAN clustering, images assigned to the noise category (`cluster = -1`) are removed before visualization. These images represent points that do not meet DBSCAN's density requirements and may not belong to a meaningful cluster. Removing them before dimensionality reduction prevents outlier images from influencing the PCA and t-SNE projections, allowing the visualization to focus on images that were assigned to identified clusters.

PCA is first applied to reduce the dimensionality of the image embeddings while retaining the most important patterns of variation within the data. The resulting lower-dimensional representation is then passed into t-distributed Stochastic Neighbor Embedding (t-SNE), which projects the embeddings into two dimensions. Unlike PCA, which focuses on preserving overall variance, t-SNE emphasizes local neighborhood relationships, making it useful for visually examining whether similar images appear near one another.

The final t-SNE visualization provides a qualitative method for evaluating DBSCAN results. Each point represents an image embedding, and colors indicate the assigned cluster. These visualizations are used to inspect cluster separation, identify possible misclustered images, and assess whether groups of visually similar individuals appear together.

### Outputs

The output of the pipeline is:

```
cluster_embeddings/
└── clusters/
    ├── dbscan_assignments.json
    └── dbscan_tsne_visualization.png
```

The `dbscan_assignments.json` file stores the predicted cluster assignment for each image.

Each record contains:

| Field | Description |
| --- | --- |
| Image identifier | Filename or identifier associated with the image. |
| Cluster | DBSCAN cluster assignment. A value of `-1` indicates a noise point. |

The visualization output provides a two-dimensional representation of the discovered clusters. Each point represents one image, and colors indicate the DBSCAN cluster assignment. These visualizations are used to evaluate cluster separation, identify possible errors, and understand patterns within the image collection.

### Limitations

DBSCAN performance depends heavily on the quality of the image embeddings and the selected parameters. Because the dataset contains a limited number (5,000 total images) of historical images with variations in pose, lighting, age, hairstyle, and image quality, visually similar images may not always correspond to the same identity.

---

## Zero-Shot Concept Labeling

