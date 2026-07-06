"""
generate_clip_embeddings.py

Generates CLIP embeddings for local JPG images using SentenceTransformer.

Outputs:
1. embeddings/<image_name>.npy
   - one embedding per image, shape (512,)

2. embeddings/global_embeddings.npy
   - all embeddings stacked together, shape (N, 512)

3. embeddings/global_embeddings.txt
   - filenames, one per line
   - line i corresponds to row i in global_embeddings.npy
"""

from sentence_transformers import SentenceTransformer
import PIL.Image
import numpy as np
import glob
import os


IMAGE_FOLDER = "sample_photos"
OUTPUT_FOLDER = "embeddings"

GLOBAL_NPY = os.path.join(OUTPUT_FOLDER, "global_embeddings.npy")
GLOBAL_TXT = os.path.join(OUTPUT_FOLDER, "global_embeddings.txt")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# load CLIP model
model = SentenceTransformer("clip-ViT-B-32")
print("Loaded model!")


def generate_embeddings(file_list):
    global_np_list = []
    global_file_list = []

    for i in range(len(file_list)):
        local_fp = file_list[i]

        filename = os.path.basename(local_fp)
        base_filename = os.path.splitext(filename)[0]

        npy_filepath = os.path.join(OUTPUT_FOLDER, base_filename + ".npy")

        try:
            image = PIL.Image.open(local_fp, mode="r").convert("RGB")

            # embedding shape should be (512,)
            embedding = model.encode(image)

            # save individual image embedding
            np.save(npy_filepath, np.array(embedding))

            # collect for global stacked file
            global_np_list.append(embedding)
            global_file_list.append(base_filename)

            if i % 1000 == 0:
                print(i)

        except Exception as e:
            print(f"Skipping {filename}: {e}")

    # stack all embeddings into shape (N, 512)
    global_np_array = np.array(global_np_list)

    # save global embeddings matrix
    np.save(GLOBAL_NPY, global_np_array)

    # save matching filenames
    with open(GLOBAL_TXT, "w", encoding="utf-8") as f:
        for filename in global_file_list:
            f.write(f"{filename}\n")

    print("Saved individual embeddings")
    print(f"Saved global embeddings: {GLOBAL_NPY}")
    print(f"Global embedding shape: {global_np_array.shape}")
    print(f"Saved filename list: {GLOBAL_TXT}")


if __name__ == "__main__":

    files = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
    files += glob.glob(os.path.join(IMAGE_FOLDER, "*.jpeg"))
    files.sort()

    print("DONE GLOBBING")
    print(f"Found {len(files)} images")

    generate_embeddings(files)

    print("DONE EMBEDDINGS")