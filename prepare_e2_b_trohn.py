"""
Prepare a small CrossAlign E2-B pilot from the locally cached COCO images
and the public TROHN-Img dataset.

What this script does:
1) Reuses e2_data/coco_images/*.jpg (no full COCO download).
2) Finds the matching TROHN-Img rows through the Hugging Face dataset-viewer
   /filter API, rather than downloading the 9.71 GB dataset.
3) Downloads only each matched row's negative_image.
4) Excludes E2-A validation image IDs to avoid reusing that held-out set.
5) Extracts frozen OpenCLIP ViT-B/32 features using the same LAION checkpoint
   used by the existing CrossAlign project.
6) Saves e2_data/e2_b_trohn_features.pt for train_e2_b.py.

Important:
- This is a pilot built from the 416 locally cached candidate images, not the
  full 236,856-row TROHN-Img training split.
- The dataset-viewer filter index may be partial for datasets larger than 5 GB.
  The script reports this and records unmatched IDs; it never silently claims
  that all local images were retrieved.
"""

import csv
import random
import re
import time
from io import BytesIO
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "e2_data"
COCO_DIR = DATA / "coco_images"
NEG_DIR = DATA / "trohn_negative_images"
PAIR_CSV = DATA / "e2_b_trohn_pairs.csv"
MISSING_CSV = DATA / "e2_b_trohn_missing_ids.csv"
FEATURES_PATH = DATA / "e2_b_trohn_features.pt"

DATASET = "imirandam/TROHN-Img"
CONFIG = "default"
SPLIT = "train"
FILTER_URL = "https://datasets-server.huggingface.co/filter"

SEED = 42
API_BATCH_IDS = 12
API_MAX_ROWS = 100
IMAGE_BATCH_SIZE = 16
MIN_PAIRS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HEADERS = {"User-Agent": "CrossAlign-E2B/1.0"}


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_expected_pairs():
    """Use already-existing candidate metadata to prefer the exact old pair."""
    sources = [
        DATA / "trohn_clip_scored.csv",
        DATA / "e2_binding_scored.csv",
        DATA / "e2_binding_fullscan.csv",
        DATA / "trohn_train_candidates.csv",
    ]
    expected = {}
    for path in sources:
        for row in read_csv_rows(path):
            image_id = str(row.get("image_id", "")).strip()
            positive = str(row.get("positive", row.get("caption", ""))).strip()
            negative = str(row.get("negative", row.get("negative_caption", ""))).strip()
            if image_id and positive and negative:
                expected.setdefault(image_id, [])
                pair = (positive, negative)
                if pair not in expected[image_id]:
                    expected[image_id].append(pair)
    return expected


def load_e2a_validation_ids():
    path = DATA / "e2_a_case_diagnostic.csv"
    rows = read_csv_rows(path)
    if not rows:
        print("WARNING: e2_a_case_diagnostic.csv not found; "
              "cannot automatically exclude E2-A validation IDs.")
        return set()
    return {
        str(r["image_id"]).strip()
        for r in rows
        if str(r.get("split", "")).strip().lower() in {"val", "valid", "validation"}
    }


def verify_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def query_filter_batch(image_ids):
    where = " OR ".join(
        f'"image_id"=\'{image_id}\''
        for image_id in image_ids
    )
    params = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "where": where,
        "offset": 0,
        "length": API_MAX_ROWS,
    }

    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(
                FILTER_URL,
                params=params,
                headers=HEADERS,
                timeout=(15, 90),
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(attempt + 1)

    raise RuntimeError(
        f"Hugging Face filter request failed for {len(image_ids)} IDs: "
        f"{type(last_error).__name__}: {last_error}"
    )


def download_negative_image(url, destination):
    if destination.exists() and verify_image(destination):
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=(15, 90),
            )
            response.raise_for_status()
            data = response.content

            with Image.open(BytesIO(data)) as im:
                im.verify()

            destination.write_bytes(data)
            return True
        except Exception as exc:
            last_error = exc
            time.sleep(attempt + 1)

    print(
        "Negative-image download failed:",
        destination.name,
        type(last_error).__name__,
        str(last_error)[:180],
    )
    return False


def main():
    DATA.mkdir(exist_ok=True)
    NEG_DIR.mkdir(exist_ok=True)

    if not COCO_DIR.exists():
        raise FileNotFoundError(f"COCO image cache not found: {COCO_DIR}")

    expected = load_expected_pairs()
    local_ids = sorted(
        p.name for p in COCO_DIR.glob("*.jpg")
        if re.fullmatch(r"\d{12}\.jpg", p.name)
    )

    # Only use local images that can be traced to the previously prepared TROHN
    # candidate metadata.
    local_ids = [image_id for image_id in local_ids if image_id in expected]
    if not local_ids:
        raise RuntimeError(
            "No local COCO image IDs matched the existing TROHN candidate CSVs."
        )

    e2a_val_ids = load_e2a_validation_ids()
    before = len(local_ids)
    local_ids = [x for x in local_ids if x not in e2a_val_ids]

    print("Local COCO images with TROHN metadata:", before)
    print("Excluded E2-A validation IDs:", before - len(local_ids))
    print("IDs to retrieve from TROHN-Img:", len(local_ids))

    rows_by_id = {}
    partial_flags = []

    for start in range(0, len(local_ids), API_BATCH_IDS):
        batch_ids = local_ids[start:start + API_BATCH_IDS]
        payload = query_filter_batch(batch_ids)
        partial_flags.append(bool(payload.get("partial", False)))

        for item in payload.get("rows", []):
            row = item.get("row", {})
            image_id = str(row.get("image_id", "")).strip()
            if image_id in batch_ids:
                rows_by_id.setdefault(image_id, []).append(row)

        done = min(start + len(batch_ids), len(local_ids))
        print(f"Queried IDs: {done}/{len(local_ids)}")

    records = []
    missing = []

    for image_id in local_ids:
        candidates = rows_by_id.get(image_id, [])
        if not candidates:
            missing.append({
                "image_id": image_id,
                "reason": "NOT_RETURNED_BY_DATASET_VIEWER_FILTER",
            })
            continue

        # Prefer the exact positive/negative pair used by the earlier candidate
        # scoring scripts. If multiple TROHN captions exist for this image ID
        # and none matches, use the first official TROHN-Img row for that image.
        expected_pairs = expected.get(image_id, [])
        selected = None
        for candidate in candidates:
            pair = (
                str(candidate.get("caption", "")).strip(),
                str(candidate.get("negative_caption", "")).strip(),
            )
            if pair in expected_pairs:
                selected = candidate
                break
        if selected is None:
            selected = candidates[0]

        negative_image = selected.get("negative_image")
        image_url = negative_image.get("src") if isinstance(negative_image, dict) else None
        if not image_url:
            missing.append({
                "image_id": image_id,
                "reason": "NEGATIVE_IMAGE_URL_MISSING",
            })
            continue

        coco_path = COCO_DIR / image_id
        if not verify_image(coco_path):
            missing.append({
                "image_id": image_id,
                "reason": "LOCAL_COCO_IMAGE_INVALID",
            })
            continue

        negative_path = NEG_DIR / image_id
        if not download_negative_image(image_url, negative_path):
            missing.append({
                "image_id": image_id,
                "reason": "NEGATIVE_IMAGE_DOWNLOAD_FAILED",
            })
            continue

        records.append({
            "image_id": image_id,
            "positive": str(selected.get("caption", "")).strip(),
            "negative": str(selected.get("negative_caption", "")).strip(),
            "coco_image_path": str(coco_path),
            "negative_image_path": str(negative_path),
        })

    with MISSING_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "reason"])
        writer.writeheader()
        writer.writerows(missing)

    if not records:
        raise RuntimeError(
            "No complete TROHN-Img pairs were retrieved. "
            f"See {MISSING_CSV} and check Hugging Face access."
        )

    if len(records) < MIN_PAIRS:
        raise RuntimeError(
            f"Only {len(records)} complete pairs were retrieved "
            f"(minimum pilot size is {MIN_PAIRS}). "
            f"See {MISSING_CSV}. No feature extraction was started."
        )

    # Deterministic 80/20 split at image level.
    order = list(range(len(records)))
    random.Random(SEED).shuffle(order)
    val_count = max(1, round(len(order) * 0.2))
    val_indices = sorted(order[:val_count])
    train_indices = sorted(order[val_count:])
    split_map = {i: "val" for i in val_indices}
    split_map.update({i: "train" for i in train_indices})

    for i, row in enumerate(records):
        row["split"] = split_map[i]

    with PAIR_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print("\n=== TROHN-Img Pair Retrieval ===")
    print("Complete pairs:", len(records))
    print("Train:", len(train_indices))
    print("Validation:", len(val_indices))
    print("Dataset viewer partial index:", any(partial_flags))
    print("Missing IDs:", len(missing))
    print("Pair CSV:", PAIR_CSV)
    print("Missing report:", MISSING_CSV)

    # Frozen CLIP feature extraction. This matches the project's E0/E1 backbone.
    print("\nLoading OpenCLIP ViT-B/32 (LAION checkpoint) on", DEVICE)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )
    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    original_features = []
    negative_image_features = []
    positive_features = []
    negative_text_features = []

    with torch.inference_mode():
        for start in range(0, len(records), IMAGE_BATCH_SIZE):
            batch = records[start:start + IMAGE_BATCH_SIZE]

            original_tensors = []
            negative_tensors = []
            captions = []

            for row in batch:
                with Image.open(row["coco_image_path"]) as im:
                    original_tensors.append(preprocess(im.convert("RGB")))
                with Image.open(row["negative_image_path"]) as im:
                    negative_tensors.append(preprocess(im.convert("RGB")))

                captions.extend([row["positive"], row["negative"]])

            original_batch = torch.stack(original_tensors).to(DEVICE)
            negative_batch = torch.stack(negative_tensors).to(DEVICE)
            image_batch = torch.cat([original_batch, negative_batch], dim=0)

            tokens = tokenizer(captions).to(DEVICE)

            image_f = F.normalize(
                model.encode_image(image_batch).float(),
                dim=-1,
            ).cpu()

            text_f = F.normalize(
                model.encode_text(tokens).float(),
                dim=-1,
            ).cpu()

            n = len(batch)
            original_features.append(image_f[:n])
            negative_image_features.append(image_f[n:])
            positive_features.append(text_f[0::2])
            negative_text_features.append(text_f[1::2])

            print(f"Encoded: {min(start + n, len(records))}/{len(records)}")

    cache = {
        "image_features": torch.cat(original_features).float(),
        "negative_image_features": torch.cat(negative_image_features).float(),
        "positive_features": torch.cat(positive_features).float(),
        "negative_features": torch.cat(negative_text_features).float(),
        "image_ids": [r["image_id"] for r in records],
        "train_indices": torch.tensor(train_indices, dtype=torch.long),
        "val_indices": torch.tensor(val_indices, dtype=torch.long),
        "pretrained": "laion2b_s34b_b79k",
        "source": "TROHN-Img local-cache pilot",
        "dataset_viewer_partial_index": any(partial_flags),
    }

    torch.save(cache, FEATURES_PATH)

    print("\n=== Feature Cache Saved ===")
    print("Original images:", tuple(cache["image_features"].shape))
    print("Negative images:", tuple(cache["negative_image_features"].shape))
    print("Positive texts:", tuple(cache["positive_features"].shape))
    print("Negative texts:", tuple(cache["negative_features"].shape))
    print("Saved:", FEATURES_PATH)


if __name__ == "__main__":
    main()
