
import csv
import random
import re
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

BASE = Path(__file__).resolve().parent
DATA = BASE / "e2_data"
SOURCE = DATA / "e2_binding_scored.csv"
FEATURES = DATA / "e2_attr_features.pt"
FILTER_REPORT = DATA / "e2_attr_filter_report.csv"
CHECKPOINT = BASE / "clip_adapter_e1.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
SEED = 42

# 保守过滤：排除容易产生语言捷径的动物属性表达。
# 这不代表其他候选均已通过人工验证。
SHORTCUT_PATTERN = re.compile(
    r"\b(?:blue|green|pink|purple|wooden)\s+"
    r"(?:cats?|dogs?)\b",
    re.IGNORECASE
)

# 1. 读取此前评分的225条候选
with SOURCE.open(
    encoding="utf-8-sig", newline=""
) as f:
    rows = list(csv.DictReader(f))

kept = []
report = []

for row in rows:
    if SHORTCUT_PATTERN.search(row["negative"]):
        reason = "POTENTIAL_LANGUAGE_SHORTCUT"
    else:
        reason = "PASS_TEXT_HEURISTIC"

    report.append({
        **row,
        "filter_result": reason
    })

    if reason == "PASS_TEXT_HEURISTIC":
        kept.append(row)

if not kept:
    raise RuntimeError("No candidates survived filtering.")

# 保存过滤记录，保证数据可追溯
with FILTER_REPORT.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(report[0].keys())
    )
    writer.writeheader()
    writer.writerows(report)

print("All candidates:", len(rows))
print("Excluded:", len(rows) - len(kept))
print("Remaining candidates:", len(kept))
print("Device:", DEVICE)

# 2. 加载与E0、E1相同的预训练CLIP
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image_vectors = []
positive_vectors = []
negative_vectors = []
valid_rows = []

# 3. 提取正图片、正确文本和负文本特征
with torch.inference_mode():

    for start in range(0, len(kept), BATCH_SIZE):
        batch = kept[start:start + BATCH_SIZE]

        tensors = []
        accepted = []

        for row in batch:
            image_path = Path(
                row.get("image_path")
                or f"e2_data/coco_images/{row['image_id']}"
            )

            if not image_path.is_absolute():
                image_path = BASE / image_path

            try:
                with Image.open(image_path) as im:
                    tensors.append(
                        preprocess(im.convert("RGB"))
                    )
                accepted.append(row)

            except (OSError, ValueError) as exc:
                print(
                    "Skipping:",
                    row["image_id"],
                    str(exc)[:100]
                )

        if not accepted:
            continue

        images = torch.stack(tensors).to(DEVICE)

        captions = [
            text
            for row in accepted
            for text in (
                row["positive"],
                row["negative"]
            )
        ]

        tokens = tokenizer(captions).to(DEVICE)

        image_f = F.normalize(
            model.encode_image(images).float(),
            dim=-1
        ).cpu()

        text_f = F.normalize(
            model.encode_text(tokens).float(),
            dim=-1
        ).cpu()

        image_vectors.append(image_f)
        positive_vectors.append(text_f[0::2])
        negative_vectors.append(text_f[1::2])

        valid_rows.extend(accepted)

        print(
            f"Encoded: {len(valid_rows)}/{len(kept)}"
        )

if not valid_rows:
    raise RuntimeError("No usable image-caption pairs.")

# 4. 按图片划分专项训练和验证子集
n = len(valid_rows)

indices = list(range(n))
random.Random(SEED).shuffle(indices)

val_count = max(1, round(n * 0.2))

train_indices = indices[val_count:]
val_indices = indices[:val_count]

# 5. 缓存所有特征
torch.save(
    {
        "image_features": torch.cat(image_vectors),
        "positive_features": torch.cat(positive_vectors),
        "negative_features": torch.cat(negative_vectors),
        "image_ids": [
            row["image_id"] for row in valid_rows
        ],
        "clip_margins": torch.tensor([
            float(row["clip_margin"])
            for row in valid_rows
        ]),
        "train_indices": torch.tensor(
            train_indices, dtype=torch.long
        ),
        "val_indices": torch.tensor(
            val_indices, dtype=torch.long
        ),
        "pretrained": "laion2b_s34b_b79k",
        "label_status": "UNVERIFIED_CANDIDATES"
    },
    FEATURES
)

print("\n=== E2-A Feature Preparation ===")
print("Usable candidates:", n)
print("Train:", len(train_indices))
print("Val:", len(val_indices))
print(
    "Image features:",
    tuple(torch.cat(image_vectors).shape)
)
print(
    "Positive features:",
    tuple(torch.cat(positive_vectors).shape)
)
print(
    "Negative features:",
    tuple(torch.cat(negative_vectors).shape)
)
print("Saved:", FEATURES)

# 6. 检查现有E1权重格式，避免猜测Adapter结构
print("\n=== E1 Checkpoint Inspection ===")
print("Checkpoint exists:", CHECKPOINT.exists())

if CHECKPOINT.exists():
    try:
        ckpt = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True
        )

        print("Checkpoint type:", type(ckpt).__name__)

        if isinstance(ckpt, dict):
            print("Top-level keys:", list(ckpt)[:30])

            for key, value in list(ckpt.items())[:10]:
                if isinstance(value, torch.Tensor):
                    print(key, tuple(value.shape))

                elif isinstance(value, dict):
                    print(
                        key,
                        "nested keys:",
                        list(value)[:12]
                    )

                else:
                    print(
                        key,
                        "type:",
                        type(value).__name__
                    )

    except Exception as exc:
        print(
            "Checkpoint inspection error:",
            type(exc).__name__,
            str(exc)[:300]
        )
