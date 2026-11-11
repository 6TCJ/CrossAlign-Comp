"""
CrossAlign E2-B data preparation (v2).

Fixes the previous version's fragile Hugging Face /filter OR query:
- Reuses the already-working streaming metadata scan used by mine_e2_full.py.
- Reads only image_id/caption/negative_caption during the scan.
- Records the exact streaming row index for each matched TROHN-Img example.
- Retrieves only the selected row's negative_image URL through the dataset-viewer
  /rows endpoint, then downloads only that negative image.
- Reuses local COCO originals and excludes the E2-A held-out validation IDs.

Run from the project root:
    python prepare_e2_b_trohn_v2.py
"""

import csv
import random
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
import open_clip
from datasets import load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "e2_data"
COCO_DIR = DATA / "coco_images"
NEG_DIR = DATA / "trohn_negative_images"

PAIR_CSV = DATA / "e2_b_trohn_pairs.csv"
MISSING_CSV = DATA / "e2_b_trohn_missing_ids.csv"
FEATURES_PATH = DATA / "e2_b_trohn_features.pt"

DATASET = "imirandam/TROHN-Img"
SPLIT = "train"
ROWS_API = "https://datasets-server.huggingface.co/rows"

SEED = 42
IMAGE_BATCH_SIZE = 16
DOWNLOAD_WORKERS = 6
MIN_PAIRS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HEADERS = {"User-Agent": "CrossAlign-E2B/2.0"}


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_expected_pairs():
    """Collect exact positive/negative text pairs already used in this project."""
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
            positive = str(
                row.get("positive") or row.get("caption") or ""
            ).strip()
            negative = str(
                row.get("negative") or row.get("negative_caption") or ""
            ).strip()

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
        print(
            "WARNING: e2_a_case_diagnostic.csv not found. "
            "No E2-A validation IDs can be excluded automatically."
        )
        return set()

    return {
        str(row.get("image_id", "")).strip()
        for row in rows
        if str(row.get("split", "")).strip().lower()
        in {"val", "valid", "validation"}
    }


def verify_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def scan_trohn_metadata(target_ids, expected_pairs):
    """
    Scan the same three text/ID columns used by the already-completed
    mine_e2_full.py. The generated-image column is deliberately not selected.
    """
    print("\nLoading TROHN-Img metadata in streaming mode...")
    print("Selected columns: image_id, caption, negative_caption")
    print("No negative_image column is read during this scan.")

    dataset = load_dataset(
        DATASET,
        split=SPLIT,
        streaming=True,
        columns=["image_id", "caption", "negative_caption"],
    )

    exact_matches = {}
    fallback_matches = {}
    scanned = 0

    for row_idx, row in enumerate(dataset):
        scanned += 1
        image_id = str(row.get("image_id", "")).strip()

        if image_id in target_ids:
            positive = str(row.get("caption", "")).strip()
            negative = str(row.get("negative_caption", "")).strip()

            record = {
                "row_idx": int(row_idx),
                "image_id": image_id,
                "positive": positive,
                "negative": negative,
                "pair_match_status": "EXACT_PROJECT_PAIR",
            }

            if image_id not in fallback_matches:
                fallback = dict(record)
                fallback["pair_match_status"] = "SAME_IMAGE_ID_FALLBACK"
                fallback_matches[image_id] = fallback

            if (
                image_id not in exact_matches
                and (positive, negative) in expected_pairs.get(image_id, [])
            ):
                exact_matches[image_id] = record

        if scanned % 10000 == 0:
            print(
                f"Scanned metadata: {scanned} | "
                f"Exact matched IDs: {len(exact_matches)}/{len(target_ids)}"
            )

        if len(exact_matches) == len(target_ids):
            break

    # Prefer exact pairs; use same-image-ID fallback only when no exact pair
    # exists in the TROHN-Img train split.
    selected = dict(fallback_matches)
    selected.update(exact_matches)

    missing = sorted(target_ids - set(selected))
    print("\n=== Metadata matching ===")
    print("Rows scanned:", scanned)
    print("Exact project-pair matches:", len(exact_matches))
    print(
        "Same-image-ID fallback matches:",
        sum(
            r["pair_match_status"] == "SAME_IMAGE_ID_FALLBACK"
            for r in selected.values()
        ),
    )
    print("Unmatched image IDs:", len(missing))

    return selected, missing


def get_negative_image_url(row_idx, expected_image_id):
    params = {
        "dataset": DATASET,
        "config": "default",
        "split": SPLIT,
        "offset": int(row_idx),
        "length": 1,
    }

    last_error = None
    for attempt in range(4):
        try:
            response = requests.get(
                ROWS_API,
                params=params,
                headers=HEADERS,
                timeout=(15, 90),
            )
            response.raise_for_status()
            payload = response.json()

            returned_rows = payload.get("rows", [])
            if not returned_rows:
                raise RuntimeError(
                    f"Rows API returned no data for row_idx={row_idx}"
                )

            item = returned_rows[0]
            row = item.get("row", {})
            returned_idx = int(item.get("row_idx", -1))

            if returned_idx != int(row_idx):
                raise RuntimeError(
                    f"Rows API index mismatch: expected {row_idx}, "
                    f"received {returned_idx}"
                )

            returned_image_id = str(row.get("image_id", "")).strip()
            if returned_image_id != expected_image_id:
                raise RuntimeError(
                    f"Rows API image_id mismatch at row_idx={row_idx}: "
                    f"expected {expected_image_id}, received {returned_image_id}"
                )

            image_value = row.get("negative_image")
            if isinstance(image_value, dict):
                image_url = image_value.get("src")
            elif isinstance(image_value, str):
                image_url = image_value
            else:
                image_url = None

            if not image_url:
                raise RuntimeError(
                    f"negative_image URL missing at row_idx={row_idx}"
                )

            return image_url

        except Exception as exc:
            last_error = exc
            time.sleep(attempt + 1)

    raise RuntimeError(
        f"Could not retrieve negative_image URL for row_idx={row_idx}: "
        f"{type(last_error).__name__}: {last_error}"
    )


def download_one_negative(record):
    image_id = record["image_id"]
    destination = NEG_DIR / image_id

    if destination.exists() and verify_image(destination):
        return record, destination, ""

    try:
        image_url = get_negative_image_url(record["row_idx"], image_id)

        last_error = None
        for attempt in range(4):
            try:
                response = requests.get(
                    image_url,
                    headers=HEADERS,
                    timeout=(15, 90),
                )
                response.raise_for_status()

                data = response.content
                with Image.open(BytesIO(data)) as im:
                    im.verify()

                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                return record, destination, ""

            except Exception as exc:
                last_error = exc
                time.sleep(attempt + 1)

        raise RuntimeError(
            f"Image download failed: {type(last_error).__name__}: {last_error}"
        )

    except Exception as exc:
        return record, None, f"{type(exc).__name__}: {exc}"


def extract_features(records):
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
            image_batch = torch.cat(
                [original_batch, negative_batch], dim=0
            )
            tokens = tokenizer(captions).to(DEVICE)

            image_f = F.normalize(
                model.encode_image(image_batch).float(), dim=-1
            ).cpu()
            text_f = F.normalize(
                model.encode_text(tokens).float(), dim=-1
            ).cpu()

            n = len(batch)
            original_features.append(image_f[:n])
            negative_image_features.append(image_f[n:])
            positive_features.append(text_f[0::2])
            negative_text_features.append(text_f[1::2])

            print(
                f"Feature extraction: "
                f"{min(start + n, len(records))}/{len(records)}"
            )

    return {
        "image_features": torch.cat(original_features).float(),
        "negative_image_features": torch.cat(
            negative_image_features
        ).float(),
        "positive_features": torch.cat(positive_features).float(),
        "negative_features": torch.cat(negative_text_features).float(),
    }


def main():
    DATA.mkdir(exist_ok=True)
    NEG_DIR.mkdir(exist_ok=True)

    if not COCO_DIR.exists():
        raise FileNotFoundError(f"COCO image cache not found: {COCO_DIR}")

    expected = load_expected_pairs()
    local_ids = sorted(
        path.name
        for path in COCO_DIR.glob("*.jpg")
        if path.is_file()
    )
    local_ids = [image_id for image_id in local_ids if image_id in expected]

    if not local_ids:
        raise RuntimeError(
            "No local COCO images matched the existing TROHN candidate CSVs."
        )

    validation_ids = load_e2a_validation_ids()
    before = len(local_ids)
    target_ids = set(local_ids) - validation_ids

    print("Local COCO images with TROHN metadata:", before)
    print("Excluded E2-A validation IDs:", before - len(target_ids))
    print("IDs to retrieve from TROHN-Img:", len(target_ids))

    matched, metadata_missing = scan_trohn_metadata(
        target_ids, expected
    )

    if not matched:
        raise RuntimeError(
            "No TROHN-Img rows matched the local image IDs. "
            "No negative images were downloaded."
        )

    selected_records = [matched[key] for key in sorted(matched)]

    print("\nRetrieving only the matched negative images...")
    downloaded = []
    failures = []

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for record, path, error in pool.map(
            download_one_negative, selected_records
        ):
            if path is None:
                failures.append({
                    "image_id": record["image_id"],
                    "row_idx": record["row_idx"],
                    "reason": error,
                })
            else:
                item = dict(record)
                item["coco_image_path"] = str(COCO_DIR / record["image_id"])
                item["negative_image_path"] = str(path)
                downloaded.append(item)

            print(
                f"Negative images retrieved: "
                f"{len(downloaded) + len(failures)}/{len(selected_records)}"
            )

    for image_id in metadata_missing:
        failures.append({
            "image_id": image_id,
            "row_idx": "",
            "reason": "IMAGE_ID_NOT_FOUND_IN_TROHN_TRAIN",
        })

    with MISSING_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "row_idx", "reason"]
        )
        writer.writeheader()
        writer.writerows(failures)

    if len(downloaded) < MIN_PAIRS:
        raise RuntimeError(
            f"Only {len(downloaded)} complete pairs are available; "
            f"at least {MIN_PAIRS} are required for this pilot. "
            f"See {MISSING_CSV}. Feature extraction was not started."
        )

    random.Random(SEED).shuffle(downloaded)
    val_count = max(1, round(0.2 * len(downloaded)))
    val_indices = list(range(val_count))
    train_indices = list(range(val_count, len(downloaded)))

    for index, row in enumerate(downloaded):
        row["split"] = "val" if index in set(val_indices) else "train"

    with PAIR_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(downloaded[0].keys()))
        writer.writeheader()
        writer.writerows(downloaded)

    print("\n=== TROHN-Img Pilot Data ===")
    print("Complete image-text pairs:", len(downloaded))
    print("Train:", len(train_indices))
    print("Validation:", len(val_indices))
    print("Exact pair matches:", sum(
        r["pair_match_status"] == "EXACT_PROJECT_PAIR"
        for r in downloaded
    ))
    print("Fallback matches:", sum(
        r["pair_match_status"] == "SAME_IMAGE_ID_FALLBACK"
        for r in downloaded
    ))
    print("Missing / failed:", len(failures))
    print("Pair CSV:", PAIR_CSV)
    print("Missing report:", MISSING_CSV)

    features = extract_features(downloaded)
    features.update({
        "image_ids": [row["image_id"] for row in downloaded],
        "train_indices": torch.tensor(train_indices, dtype=torch.long),
        "val_indices": torch.tensor(val_indices, dtype=torch.long),
        "pretrained": "laion2b_s34b_b79k",
        "source": "TROHN-Img local-cache pilot, streaming metadata + rows API",
    })

    torch.save(features, FEATURES_PATH)

    print("\n=== Feature Cache Saved ===")
    for key in (
        "image_features",
        "negative_image_features",
        "positive_features",
        "negative_features",
    ):
        print(key, tuple(features[key].shape))
    print("Saved:", FEATURES_PATH)


if __name__ == "__main__":
    main()
