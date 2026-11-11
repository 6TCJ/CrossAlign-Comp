"""
CrossAlign E2-B full-data preflight.

Purpose:
1. Verify the saved 373-row Pilot cache and its backup.
2. Stream ONLY TROHN-Img train metadata (image_id/caption/negative_caption).
3. Save a full train manifest and unique COCO image-ID inventory.
4. Count how many original COCO images are already cached locally.

This does NOT download negative images, COCO images, or the 9.71 GB dataset.
It does NOT overwrite the Pilot feature cache or train any model.

Run from the CrossAlign project root:
    python inspect_trohn_full_manifest.py
"""

import csv
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "e2_data"
COCO_DIR = DATA / "coco_images"

PILOT_CACHE = DATA / "e2_b_trohn_features.pt"
PILOT_BACKUP = DATA / "e2_b_trohn_pilot_features.pt"

TRAIN_MANIFEST = DATA / "trohn_img_train_manifest.csv"
UNIQUE_IDS_CSV = DATA / "trohn_img_train_unique_image_ids.csv"

DATASET_ID = "imirandam/TROHN-Img"
EXPECTED_TRAIN_ROWS = 236_856
PROGRESS_EVERY = 10_000


def verify_pilot_backup():
    print("=== 1. Verify Pilot cache and backup ===")

    if not PILOT_CACHE.exists():
        raise FileNotFoundError(f"Pilot cache not found: {PILOT_CACHE}")
    if not PILOT_BACKUP.exists():
        raise FileNotFoundError(
            f"Pilot backup not found: {PILOT_BACKUP}. "
            "Run the Copy-Item command first."
        )

    print("Original cache size (MB):", round(PILOT_CACHE.stat().st_size / 1024**2, 3))
    print("Backup cache size (MB):", round(PILOT_BACKUP.stat().st_size / 1024**2, 3))

    original = torch.load(PILOT_CACHE, map_location="cpu", weights_only=True)
    backup = torch.load(PILOT_BACKUP, map_location="cpu", weights_only=True)

    required = (
        "image_features",
        "negative_image_features",
        "positive_features",
        "negative_features",
    )

    for key in required:
        if key not in original or key not in backup:
            raise RuntimeError(f"Missing tensor key in cache/backup: {key}")
        if original[key].shape != backup[key].shape:
            raise RuntimeError(
                f"Shape mismatch for {key}: "
                f"{tuple(original[key].shape)} vs {tuple(backup[key].shape)}"
            )
        if not torch.equal(original[key], backup[key]):
            raise RuntimeError(f"Backup tensor differs from original: {key}")
        print(f"{key}: {tuple(original[key].shape)} | identical")

    n = original["image_features"].shape[0]
    if n != 373:
        print(f"WARNING: expected 373 Pilot rows, found {n}")
    else:
        print("Pilot backup verification: PASSED (373 rows)")

    del original, backup


def build_train_manifest():
    print("\n=== 2. Stream full TROHN-Img train metadata ===")
    print("Dataset:", DATASET_ID)
    print("Columns: image_id, caption, negative_caption")
    print("The negative_image column is NOT loaded in this preflight.")

    token = get_token()
    if token:
        print("Hugging Face authentication: token found")
    else:
        print("WARNING: no Hugging Face token found; anonymous rate limits may apply")

    dataset = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
        columns=["image_id", "caption", "negative_caption"],
        token=token,
    )

    DATA.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    rows_written = 0

    with TRAIN_MANIFEST.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["row_idx", "image_id", "caption", "negative_caption"],
        )
        writer.writeheader()

        for row_idx, row in enumerate(dataset):
            image_id = str(row.get("image_id", "")).strip()
            caption = str(row.get("caption", "")).strip()
            negative_caption = str(row.get("negative_caption", "")).strip()

            if not image_id or not caption or not negative_caption:
                continue

            writer.writerow({
                "row_idx": row_idx,
                "image_id": image_id,
                "caption": caption,
                "negative_caption": negative_caption,
            })
            counts[image_id] += 1
            rows_written += 1

            if rows_written % PROGRESS_EVERY == 0:
                f.flush()
                print(
                    f"Rows written: {rows_written:,} | "
                    f"Unique COCO image IDs: {len(counts):,}"
                )

    unique_ids = sorted(counts)
    local_ids = {
        p.name for p in COCO_DIR.glob("*.jpg")
        if p.is_file()
    } if COCO_DIR.exists() else set()

    with UNIQUE_IDS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "trohn_rows", "local_coco_exists"],
        )
        writer.writeheader()
        for image_id in unique_ids:
            writer.writerow({
                "image_id": image_id,
                "trohn_rows": counts[image_id],
                "local_coco_exists": image_id in local_ids,
            })

    local_overlap = len(set(unique_ids) & local_ids)
    missing_originals = len(unique_ids) - local_overlap

    print("\n=== 3. Full train metadata summary ===")
    print("Rows written:", f"{rows_written:,}")
    print("Expected train rows:", f"{EXPECTED_TRAIN_ROWS:,}")
    print("Unique COCO image IDs:", f"{len(unique_ids):,}")
    print("Rows beyond unique IDs:", f"{rows_written - len(unique_ids):,}")
    print("Local COCO images available:", f"{local_overlap:,}")
    print("Original COCO images still missing:", f"{missing_originals:,}")
    print("Train manifest:", TRAIN_MANIFEST)
    print("Unique-ID inventory:", UNIQUE_IDS_CSV)

    if rows_written != EXPECTED_TRAIN_ROWS:
        print(
            "WARNING: row count differs from the expected count. "
            "Keep the actual count and inspect the dataset split before training."
        )

    print("\nPreflight complete. No image files were downloaded.")


def main():
    verify_pilot_backup()
    build_train_manifest()


if __name__ == "__main__":
    main()
