"""
Resume CrossAlign E2-B TROHN-Img preparation without rescanning the dataset.

This script:
- Reuses all existing local COCO originals and downloaded negative images.
- Reads the existing e2_b_trohn_missing_ids.csv (image_id + row_idx).
- Uses a locally stored Hugging Face read token.
- Retries only missing negative-image lookups, sequentially and with backoff.
- Reconstructs the 373 image/text pairs from the project's existing CSVs.
- Extracts frozen OpenCLIP ViT-B/32 features and saves the training cache.

Before running:
    hf auth login

Do NOT paste your HF token into this file or into ChatGPT.
Run from the project root:
    python resume_e2_b_trohn.py
"""

import csv
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "e2_data"
COCO_DIR = DATA / "coco_images"
NEG_DIR = DATA / "trohn_negative_images"

OLD_MISSING_CSV = DATA / "e2_b_trohn_missing_ids.csv"
RETRY_FAILURES_CSV = DATA / "e2_b_trohn_retry_failures.csv"
PAIR_CSV = DATA / "e2_b_trohn_pairs.csv"
FEATURES_PATH = DATA / "e2_b_trohn_features.pt"

DATASET = "imirandam/TROHN-Img"
CONFIG = "default"
SPLIT = "train"
ROWS_API = "https://datasets-server.huggingface.co/rows"

SEED = 42
MIN_PAIRS_FOR_PILOT = 80
IMAGE_BATCH_SIZE = 16
REQUEST_PAUSE_SECONDS = 0.6
MAX_RETRIES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

session = requests.Session()
session.headers.update({"User-Agent": "CrossAlign-E2B-resume/1.0"})


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_expected_pairs():
    # Keep the same source priority as prepare_e2_b_trohn_v2.py.
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
    rows = read_csv_rows(DATA / "e2_a_case_diagnostic.csv")
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


def retry_wait_seconds(response, attempt):
    """Honor Retry-After / RateLimit headers; otherwise use bounded backoff."""
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after:
        try:
            return max(1.0, min(float(retry_after), 600.0))
        except ValueError:
            try:
                retry_date = parsedate_to_datetime(retry_after)
                if retry_date.tzinfo is None:
                    retry_date = retry_date.replace(tzinfo=timezone.utc)
                seconds = (retry_date - datetime.now(timezone.utc)).total_seconds()
                return max(1.0, min(seconds, 600.0))
            except Exception:
                pass

    rate_limit = response.headers.get("RateLimit", "")
    match = re.search(r"(?:^|[;, ])t=(\d+)", rate_limit)
    if match:
        return max(1.0, min(float(match.group(1)), 600.0))

    # No server hint: back off progressively rather than hammering the endpoint.
    return min(30.0 * (2 ** attempt), 300.0)


def get_with_backoff(url, *, headers, params=None, stream=False):
    last_response = None
    for attempt in range(MAX_RETRIES):
        response = session.get(
            url,
            headers=headers,
            params=params,
            timeout=(20, 120),
            stream=stream,
        )
        if response.status_code == 429 or response.status_code >= 500:
            last_response = response
            if attempt == MAX_RETRIES - 1:
                response.raise_for_status()
            wait = retry_wait_seconds(response, attempt)
            print(
                f"HTTP {response.status_code}; waiting {wait:.0f}s "
                f"before retry {attempt + 2}/{MAX_RETRIES}..."
            )
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response

    if last_response is not None:
        last_response.raise_for_status()
    raise RuntimeError("Request retry loop ended unexpectedly.")


def get_hf_auth_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "User-Agent": "CrossAlign-E2B-resume/1.0",
    }


def fetch_row_and_download(record, token, expected_pairs):
    image_id = record["image_id"]
    row_idx = int(record["row_idx"])
    negative_path = NEG_DIR / image_id

    if negative_path.exists() and verify_image(negative_path):
        return {
            "ok": True,
            "image_id": image_id,
            "positive": None,
            "negative": None,
            "status": "ALREADY_DOWNLOADED",
        }

    params = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": row_idx,
        "length": 1,
    }
    response = get_with_backoff(
        ROWS_API,
        headers=get_hf_auth_headers(token),
        params=params,
    )
    payload = response.json()
    rows = payload.get("rows", [])
    if not rows:
        raise RuntimeError(f"No row returned for row_idx={row_idx}")

    item = rows[0]
    returned_idx = int(item.get("row_idx", -1))
    row = item.get("row", {})

    if returned_idx != row_idx:
        raise RuntimeError(
            f"row_idx mismatch: expected {row_idx}, received {returned_idx}"
        )

    returned_id = str(row.get("image_id", "")).strip()
    if returned_id != image_id:
        raise RuntimeError(
            f"image_id mismatch at row_idx={row_idx}: "
            f"expected {image_id}, received {returned_id}"
        )

    positive = str(row.get("caption", "")).strip()
    negative = str(row.get("negative_caption", "")).strip()
    if (positive, negative) not in expected_pairs.get(image_id, []):
        raise RuntimeError(
            f"Caption pair at row_idx={row_idx} does not match project CSV "
            f"for {image_id}; refusing to silently substitute a different pair."
        )

    image_value = row.get("negative_image")
    if isinstance(image_value, dict):
        image_url = image_value.get("src")
    elif isinstance(image_value, str):
        image_url = image_value
    else:
        image_url = None

    if not image_url:
        raise RuntimeError(f"negative_image URL missing for {image_id}")

    # Do not forward the private HF token to the external image/CDN URL.
    image_response = get_with_backoff(
        image_url,
        headers={"User-Agent": "CrossAlign-E2B-resume/1.0"},
    )
    data = image_response.content
    with Image.open(BytesIO(data)) as im:
        im.verify()

    NEG_DIR.mkdir(parents=True, exist_ok=True)
    negative_path.write_bytes(data)

    return {
        "ok": True,
        "image_id": image_id,
        "positive": positive,
        "negative": negative,
        "status": "DOWNLOADED_AND_VERIFIED",
    }


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
    if not COCO_DIR.exists():
        raise FileNotFoundError(f"COCO cache not found: {COCO_DIR}")
    if not OLD_MISSING_CSV.exists():
        raise FileNotFoundError(f"Missing-ID report not found: {OLD_MISSING_CSV}")

    token = get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. First run `hf auth login` "
            "in this terminal, then run this script again."
        )

    expected = load_expected_pairs()
    if not expected:
        raise RuntimeError(
            "No candidate caption pairs found in e2_data CSV files."
        )

    validation_ids = load_e2a_validation_ids()
    local_ids = sorted(
        p.name for p in COCO_DIR.glob("*.jpg") if p.is_file()
    )
    target_ids = [
        image_id for image_id in local_ids
        if image_id in expected and image_id not in validation_ids
    ]

    missing_rows = read_csv_rows(OLD_MISSING_CSV)
    retry_by_id = {}
    for row in missing_rows:
        image_id = str(row.get("image_id", "")).strip()
        row_idx = str(row.get("row_idx", "")).strip()
        if image_id and row_idx.isdigit():
            retry_by_id[image_id] = int(row_idx)

    print("Local target IDs:", len(target_ids))
    print("Previously failed IDs:", len(retry_by_id))
    print(
        "Already valid negative images:",
        sum(
            (NEG_DIR / image_id).exists()
            and verify_image(NEG_DIR / image_id)
            for image_id in target_ids
        ),
    )
    print("Authenticated Hugging Face access: yes")
    print("Retry mode: one request at a time; no metadata rescan.")

    retry_failures = []
    verified_pairs = {}

    for number, (image_id, row_idx) in enumerate(
        sorted(retry_by_id.items()), start=1
    ):
        neg_path = NEG_DIR / image_id
        if neg_path.exists() and verify_image(neg_path):
            print(
                f"Retry {number}/{len(retry_by_id)}: "
                f"{image_id} already downloaded; skip"
            )
            time.sleep(REQUEST_PAUSE_SECONDS)
            continue

        print(
            f"Retry {number}/{len(retry_by_id)}: "
            f"{image_id} (row_idx={row_idx})"
        )
        try:
            result = fetch_row_and_download(
                {"image_id": image_id, "row_idx": row_idx},
                token,
                expected,
            )
            if result["positive"] is not None:
                verified_pairs[image_id] = (
                    result["positive"], result["negative"]
                )
            print("  OK:", result["status"])
        except Exception as exc:
            retry_failures.append({
                "image_id": image_id,
                "row_idx": row_idx,
                "reason": f"{type(exc).__name__}: {exc}",
            })
            print("  FAILED:", type(exc).__name__, str(exc)[:240])

        time.sleep(REQUEST_PAUSE_SECONDS)

    with RETRY_FAILURES_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "row_idx", "reason"]
        )
        writer.writeheader()
        writer.writerows(retry_failures)

    # Assemble all valid pairs. Existing 90 images are reused. Their exact
    # captions are reconstructed from the project candidate CSVs; newly
    # retrieved rows are checked against those same candidate pairs.
    records = []
    ambiguous_ids = []

    for image_id in target_ids:
        coco_path = COCO_DIR / image_id
        negative_path = NEG_DIR / image_id

        if not verify_image(coco_path):
            continue
        if not negative_path.exists() or not verify_image(negative_path):
            continue

        if image_id in verified_pairs:
            positive, negative = verified_pairs[image_id]
            status = "EXACT_PAIR_VERIFIED_FROM_ROWS_API"
        else:
            options = expected.get(image_id, [])
            if not options:
                continue
            if len(options) > 1:
                # Do not silently attach an arbitrary caption pair to an
                # already-downloaded image when the project CSVs disagree.
                ambiguous_ids.append(image_id)
                continue
            positive, negative = options[0]
            status = "PAIR_RECONSTRUCTED_FROM_PROJECT_CSV"

        records.append({
            "image_id": image_id,
            "positive": positive,
            "negative": negative,
            "coco_image_path": str(coco_path),
            "negative_image_path": str(negative_path),
            "pair_match_status": status,
        })

    if len(records) < MIN_PAIRS_FOR_PILOT:
        print("\n=== Retry summary ===")
        print("Complete pairs currently available:", len(records))
        print("Retry failures:", len(retry_failures))
        print("Retry failure report:", RETRY_FAILURES_CSV)
        raise RuntimeError(
            f"Only {len(records)} complete pairs are available; "
            f"{MIN_PAIRS_FOR_PILOT} are required for the pilot feature cache. "
            "Successful downloads were preserved. Do not rerun the old v2 script."
        )

    random.Random(SEED).shuffle(records)
    val_count = max(1, round(0.2 * len(records)))
    val_indices = list(range(val_count))
    train_indices = list(range(val_count, len(records)))
    val_set = set(val_indices)

    for index, row in enumerate(records):
        row["split"] = "val" if index in val_set else "train"

    with PAIR_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print("\n=== TROHN-Img Pilot Data ===")
    print("Complete pairs:", len(records))
    print("Train:", len(train_indices))
    print("Validation:", len(val_indices))
    print("Ambiguous existing-image caption IDs:", len(ambiguous_ids))
    print("Remaining retry failures:", len(retry_failures))
    print("Pair CSV:", PAIR_CSV)
    print("Retry failure report:", RETRY_FAILURES_CSV)

    features = extract_features(records)
    features.update({
        "image_ids": [row["image_id"] for row in records],
        "train_indices": torch.tensor(train_indices, dtype=torch.long),
        "val_indices": torch.tensor(val_indices, dtype=torch.long),
        "pretrained": "laion2b_s34b_b79k",
        "source": "TROHN-Img local-cache pilot; authenticated resumable retrieval",
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
