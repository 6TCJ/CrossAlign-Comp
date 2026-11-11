import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from datasets import load_dataset
from huggingface_hub import get_token
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "e2_data"
COCO_DIR = DATA / "coco_images"

TRAIN_IDS = DATA / "trohn_img_train_unique_image_ids.csv"
ALL_IDS = DATA / "trohn_img_all_unique_image_ids.csv"
FAIL_CSV = DATA / "coco_download_failures.csv"

DATASET = "imirandam/TROHN-Img"
COCO_URL = "https://images.cocodataset.org/train2017/"
WORKERS = 8


def valid_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main():
    token = get_token()

    if not token:
        raise RuntimeError("HF token missing. Run hf auth login")

    with TRAIN_IDS.open("r", encoding="utf-8-sig", newline="") as f:
        train_ids = {
            r["image_id"].strip()
            for r in csv.DictReader(f)
            if r.get("image_id")
        }

    print("Train unique COCO IDs:", len(train_ids))

    print("Scanning validation IDs...")

    val_ds = load_dataset(
        DATASET,
        split="validation",
        streaming=True,
        columns=["image_id"],
        token=token
    )

    val_ids = set()
    val_rows = 0

    for row in val_ds:
        val_ids.add(str(row["image_id"]).strip())
        val_rows += 1

        if val_rows % 10000 == 0:
            print("Validation rows:", val_rows)

    all_ids = train_ids | val_ids

    print("Validation rows:", val_rows)
    print("Validation unique IDs:", len(val_ids))
    print("Shared image IDs:", len(train_ids & val_ids))
    print("Total unique images:", len(all_ids))

    with ALL_IDS.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id"])
        w.writerows((x,) for x in sorted(all_ids))

    COCO_DIR.mkdir(parents=True, exist_ok=True)

    todo = [
        x for x in sorted(all_ids)
        if not valid_image(COCO_DIR / x)
    ]

    print("Already valid:", len(all_ids) - len(todo))
    print("Need download:", len(todo))

    def fetch(image_id):
        dst = COCO_DIR / image_id
        url = COCO_URL + image_id
        last_error = ""

        for attempt in range(6):
            try:
                r = requests.get(url, timeout=(15, 90))

                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(min(5 * (2 ** attempt), 90))
                    last_error = f"HTTP {r.status_code}"
                    continue

                r.raise_for_status()

                data = r.content

                with Image.open(BytesIO(data)) as im:
                    im.verify()

                tmp = dst.with_suffix(".jpg.part")
                tmp.write_bytes(data)
                tmp.replace(dst)

                return image_id, True, ""

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt, 30))

        return image_id, False, last_error

    failures = []
    done = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:

        for start in range(0, len(todo), 500):

            futures = [
                pool.submit(fetch, x)
                for x in todo[start:start + 500]
            ]

            for future in as_completed(futures):

                image_id, ok, reason = future.result()
                done += 1

                if not ok:
                    failures.append({
                        "image_id": image_id,
                        "reason": reason
                    })

                if done % 500 == 0 or done == len(todo):
                    print(
                        f"COCO processed: {done}/{len(todo)} "
                        f"| failures={len(failures)}"
                    )

    with FAIL_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["image_id", "reason"]
        )
        w.writeheader()
        w.writerows(failures)

    missing = [
        x for x in all_ids
        if not valid_image(COCO_DIR / x)
    ]

    print("Valid originals:", len(all_ids) - len(missing), "/", len(all_ids))

    if missing:
        raise RuntimeError(
            f"{len(missing)} images still missing. "
            "Rerun this script to resume."
        )

    print("COCO original image preparation complete.")


if __name__ == "__main__":
    main()