
import csv
import re
import time
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image

DATA_DIR = Path("e2_data")
IMAGE_DIR = DATA_DIR / "coco_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = DATA_DIR / "e2_binding_fullscan.csv"
OUTPUT = DATA_DIR / "e2_binding_scored.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 读取225条结构候选
with SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("Structural candidates:", len(rows))


# 2. 获取COCO原图，优先复用已经下载的图片
def fetch(row):
    image_id = row["image_id"]

    if not re.fullmatch(r"\d{12}\.jpg", image_id):
        return row, None, "Invalid image ID"

    path = IMAGE_DIR / image_id

    if path.exists():
        try:
            with Image.open(path) as im:
                im.verify()
            return row, path, ""
        except Exception:
            path.unlink(missing_ok=True)

    # 临时沿用此前已验证可访问的HTTP地址
    url = f"http://images.cocodataset.org/train2017/{image_id}"
    error = "unknown"

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(8, 25))
            response.raise_for_status()

            data = response.content

            with Image.open(BytesIO(data)) as im:
                if im.format != "JPEG":
                    raise ValueError("Not a JPEG")
                im.verify()

            path.write_bytes(data)
            return row, path, ""

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(attempt + 1)

    return row, None, error[:160]


print("\nDownloading/caching COCO images...")

found = []
errors = []

with ThreadPoolExecutor(max_workers=6) as pool:
    futures = [pool.submit(fetch, row) for row in rows]

    for future in as_completed(futures):
        row, path, error = future.result()

        if path is None:
            errors.append((row["image_id"], error))
        else:
            row["image_path"] = str(path)
            found.append(row)

print("Available images:", len(found))
print("Failed:", len(errors))

for image_id, error in errors[:5]:
    print("Download error:", image_id, error)

if not found:
    raise RuntimeError("No images available.")

# 3. 加载原始E0 CLIP
print("\nLoading E0 OpenCLIP on", DEVICE)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# 4. 计算每张图片对正负描述的匹配分数
results = []

with torch.inference_mode():
    for start in range(0, len(found), 16):
        batch = found[start:start + 16]

        pictures = []

        for row in batch:
            with Image.open(row["image_path"]) as im:
                pictures.append(preprocess(im.convert("RGB")))

        images = torch.stack(pictures).to(DEVICE)

        texts = [
            sentence
            for row in batch
            for sentence in (row["positive"], row["negative"])
        ]

        tokens = tokenizer(texts).to(DEVICE)

        image_features = F.normalize(
            model.encode_image(images).float(), dim=-1
        )

        text_features = F.normalize(
            model.encode_text(tokens).float(), dim=-1
        )

        text_features = text_features.reshape(len(batch), 2, -1)

        scores = (
            image_features.unsqueeze(1) * text_features
        ).sum(dim=-1).cpu().tolist()

        for row, pair in zip(batch, scores):
            positive_score, negative_score = pair

            record = dict(row)

            record.update({
                "positive_score": round(positive_score, 6),
                "negative_score": round(negative_score, 6),
                "clip_margin": round(
                    positive_score - negative_score, 6
                ),
                "label_status": "UNVERIFIED"
            })

            results.append(record)

        print(f"CLIP scored: {len(results)}/{len(found)}")

# 5. 保存结果
results.sort(key=lambda r: float(r["clip_margin"]))

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

# 6. 输出统计
negative_margin = sum(
    float(r["clip_margin"]) <= 0 for r in results
)

close_margin = sum(
    0 < float(r["clip_margin"]) < 0.02 for r in results
)

print("\n=== E2 ATTRIBUTE CANDIDATE DIAGNOSTIC ===")
print("Total scored:", len(results))
print("Margin <= 0:", negative_margin)
print("0 < Margin < 0.02:", close_margin)
print(
    "Margin >= 0.02:",
    len(results) - negative_margin - close_margin
)

print("Saved:", OUTPUT)

print("\n=== First 3 Candidates ===")

for row in results[:3]:
    print("\nImage:", row["image_id"])
    print("Positive:", row["positive"])
    print("Negative:", row["negative"])
    print("Margin:", row["clip_margin"])
