import torch
import torch.nn.functional as F
import open_clip

from pathlib import Path
from io import BytesIO
from datasets import load_dataset
from huggingface_hub import get_token
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "e2_data"
COCO_DIR = DATA / "coco_full" / "train2017"
OUT = DATA / "e2_b_full"

DATASET = "imirandam/TROHN-Img"

EXPECTED = {
    "train": 236856,
    "validation": 59214
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 16
SHARD_ROWS = 4096


def neg_pil(value):

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):

        if value.get("bytes"):
            with Image.open(BytesIO(value["bytes"])) as im:
                return im.convert("RGB").copy()

        if value.get("path"):
            with Image.open(value["path"]) as im:
                return im.convert("RGB").copy()

    raise TypeError(
        f"Cannot decode negative_image: {type(value)}"
    )


def existing_rows(folder):

    files = sorted(folder.glob("shard_*.pt"))

    total = 0

    for p in files:
        d = torch.load(
            p,
            map_location="cpu",
            weights_only=True
        )
        total += len(d["image_features"])

    return total, len(files)


def extract_split(split, model, preprocess, tokenizer, token):

    folder = OUT / split
    folder.mkdir(parents=True, exist_ok=True)

    done, shard_no = existing_rows(folder)

    expected = EXPECTED[split]

    print(f"\n{split}: cached {done}/{expected} rows")

    if done == expected:
        print(split, "already complete; skip")
        return

    if done > expected:
        raise RuntimeError("Cache contains too many rows.")

    ds = load_dataset(
        DATASET,
        split=split,
        streaming=True,
        token=token
    )

    if done:
        ds = ds.skip(done)

    keys = [
        "image_features",
        "negative_image_features",
        "positive_features",
        "negative_features"
    ]

    buffers = {k: [] for k in keys}

    ids_buf = []
    row_buf = []
    batch_rows = []

    processed = 0

    def encode(rows):

        orig_t = []
        neg_t = []
        captions = []

        for row in rows:

            image_id = str(row["image_id"]).strip()

            original_path = COCO_DIR / image_id

            if not original_path.exists():
                raise FileNotFoundError(original_path)

            with Image.open(original_path) as im:
                orig_t.append(
                    preprocess(im.convert("RGB"))
                )

            neg_t.append(
                preprocess(neg_pil(row["negative_image"]))
            )

            captions.extend([
                str(row["caption"]).strip(),
                str(row["negative_caption"]).strip()
            ])

        image_batch = torch.cat([
            torch.stack(orig_t),
            torch.stack(neg_t)
        ]).to(DEVICE)

        text_batch = tokenizer(captions).to(DEVICE)

        with torch.inference_mode():

            image_f = F.normalize(
                model.encode_image(image_batch).float(),
                dim=-1
            ).cpu()

            text_f = F.normalize(
                model.encode_text(text_batch).float(),
                dim=-1
            ).cpu()

        n = len(rows)

        return {
            "image_features": image_f[:n],
            "negative_image_features": image_f[n:],
            "positive_features": text_f[0::2],
            "negative_features": text_f[1::2]
        }

    def save_shard():

        nonlocal shard_no, buffers, ids_buf, row_buf

        if not ids_buf:
            return

        obj = {
            k: torch.cat(v, dim=0).contiguous()
            for k, v in buffers.items()
        }

        obj["image_ids"] = ids_buf

        obj["row_indices"] = torch.tensor(
            row_buf,
            dtype=torch.long
        )

        path = folder / f"shard_{shard_no:05d}.pt"

        tmp = path.with_suffix(".tmp")

        torch.save(obj, tmp)
        tmp.replace(path)

        print(
            f"{split}: saved {path.name} "
            f"({len(ids_buf)} rows)"
        )

        shard_no += 1

        buffers = {k: [] for k in keys}
        ids_buf = []
        row_buf = []

    for local_idx, row in enumerate(ds):

        global_idx = done + local_idx

        batch_rows.append((global_idx, row))

        if len(batch_rows) < BATCH:
            continue

        row_ids = [x[0] for x in batch_rows]
        rows = [x[1] for x in batch_rows]

        feat = encode(rows)

        for k in keys:
            buffers[k].append(feat[k])

        ids_buf.extend(
            str(r["image_id"]).strip()
            for r in rows
        )

        row_buf.extend(row_ids)

        processed += len(rows)
        batch_rows = []

        if len(ids_buf) >= SHARD_ROWS:
            save_shard()

        if processed % 1024 == 0:
            print(
                f"{split}: {done + processed}/{expected}"
            )

    if batch_rows:

        row_ids = [x[0] for x in batch_rows]
        rows = [x[1] for x in batch_rows]

        feat = encode(rows)

        for k in keys:
            buffers[k].append(feat[k])

        ids_buf.extend(
            str(r["image_id"]).strip()
            for r in rows
        )

        row_buf.extend(row_ids)

        processed += len(rows)

    save_shard()

    total, _ = existing_rows(folder)

    print(f"{split}: total cached rows = {total}")

    if total != expected:
        raise RuntimeError(
            f"{split} incomplete: expected {expected}, found {total}. "
            "Rerun to resume."
        )


def main():

    token = get_token()

    if not token:
        raise RuntimeError("HF token missing. Run hf auth login.")

    if not COCO_DIR.exists():
        raise FileNotFoundError(
            "Run download_coco_needed.py first."
        )

    print("Device:", DEVICE)

    print("Loading OpenCLIP ViT-B/32...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    model = model.to(DEVICE).eval()

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    extract_split(
        "train", model, preprocess, tokenizer, token
    )

    extract_split(
        "validation", model, preprocess, tokenizer, token
    )

    print("\nFULL TROHN FEATURE EXTRACTION COMPLETED.")


if __name__ == "__main__":
    main()