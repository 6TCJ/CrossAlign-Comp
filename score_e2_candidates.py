
import csv
import random
import re
import time
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image

# ========== 基础设置 ==========
random.seed(42)

DATA_DIR = Path("e2_data")
IMAGE_DIR = DATA_DIR / "coco_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "trohn_train_candidates.csv"
OUTPUT_CSV = DATA_DIR / "trohn_clip_scored.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 首次只评测200组，避免一次下载过多图片
N_SWAP = 100
N_OTHER = 100

# ========== 1. 读取候选 ==========
with open(INPUT_CSV, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

swap_rows = [
    r for r in rows
    if r["candidate_type"] == "SWAP_LIKE"
]

other_rows = [
    r for r in rows
    if r["candidate_type"] != "SWAP_LIKE"
]

selected = (
    random.sample(swap_rows, min(N_SWAP, len(swap_rows))) +
    random.sample(other_rows, min(N_OTHER, len(other_rows)))
)

print("Selected candidates:", len(selected))

# ========== 2. 下载COCO原始图片 ==========
def download_image(row):
    image_id = row["image_id"]

    # 检查图片文件名格式
    if not re.fullmatch(r"\d{12}\.jpg", image_id):
        return None

    path = IMAGE_DIR / image_id

    if path.exists():
        try:
            with Image.open(path) as img:
                img.verify()
            return path
        except Exception:
            path.unlink(missing_ok=True)

    url = (
        "http://images.cocodataset.org/"
        f"train2017/{image_id}"
    )

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                timeout=(10, 25),
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()

            with Image.open(BytesIO(response.content)) as img:
                img.verify()

            path.write_bytes(response.content)
            return path

        except Exception:
            time.sleep(attempt + 1)

    return None


print("\nDownloading COCO images...")

with ThreadPoolExecutor(max_workers=6) as executor:
    paths = list(executor.map(download_image, selected))

valid_rows = []

for row, path in zip(selected, paths):
    if path is not None:
        item = dict(row)
        item["image_path"] = str(path)
        valid_rows.append(item)

print("Images downloaded:", len(valid_rows))
print("Download failures:", len(selected) - len(valid_rows))

if not valid_rows:
    raise RuntimeError(
        "No images downloaded. Check COCO network access."
    )

# ========== 3. 加载原始CLIP ==========
print("\nLoading original CLIP...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# ========== 4. 计算正负样本得分 ==========
print("\nScoring candidate pairs...")

batch_size = 16

with torch.inference_mode():
    for start in range(0, len(valid_rows), batch_size):

        batch = valid_rows[start:start + batch_size]

        images = torch.stack([
            preprocess(
                Image.open(r["image_path"]).convert("RGB")
            )
            for r in batch
        ]).to(DEVICE)

        captions = []

        for row in batch:
            captions.append(row["positive"])
            captions.append(row["negative"])

        tokens = tokenizer(captions).to(DEVICE)

        image_features = F.normalize(
            model.encode_image(images), dim=-1
        )

        text_features = F.normalize(
            model.encode_text(tokens), dim=-1
        )

        # 每张图片对应Positive和Negative两条文字
        text_features = text_features.reshape(
            len(batch), 2, -1
        )

        scores = (
            image_features.unsqueeze(1) * text_features
        ).sum(dim=-1).cpu().tolist()

        for row, pair in zip(batch, scores):
            positive_score = pair[0]
            negative_score = pair[1]

            margin = positive_score - negative_score

            row["positive_score"] = round(
                positive_score, 5
            )
            row["negative_score"] = round(
                negative_score, 5
            )
            row["clip_margin"] = round(margin, 5)

            if margin <= 0:
                level = "MODEL_PREFERS_NEGATIVE"
            elif margin < 0.02:
                level = "CLOSE_SCORES"
            else:
                level = "MODEL_DISTINGUISHES"

            row["difficulty_signal"] = level
            row["status"] = "UNVERIFIED"

        print(
            f"Scored: "
            f"{min(start + batch_size, len(valid_rows))}"
            f"/{len(valid_rows)}"
        )

# ========== 5. 按CLIP分数差排序 ==========
valid_rows.sort(
    key=lambda r: r["clip_margin"]
)

with open(
    OUTPUT_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(valid_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(valid_rows)

# ========== 6. 输出统计 ==========
print("\n=== CLIP Hardness Analysis ===")

print("Total scored:", len(valid_rows))

for level in [
    "MODEL_PREFERS_NEGATIVE",
    "CLOSE_SCORES",
    "MODEL_DISTINGUISHES"
]:
    count = sum(
        row["difficulty_signal"] == level
        for row in valid_rows
    )
    print(f"{level}: {count}")

print("\n=== First 5 Hardest Candidates ===")

for row in valid_rows[:5]:
    print("\nImage:", row["image_id"])
    print("Positive:", row["positive"])
    print("Negative:", row["negative"])
    print("Type:", row["candidate_type"])
    print("Positive score:", row["positive_score"])
    print("Negative score:", row["negative_score"])
    print("Margin:", row["clip_margin"])
    print("Signal:", row["difficulty_signal"])

print("\nSaved:", OUTPUT_CSV)
