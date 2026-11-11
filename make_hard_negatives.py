
import csv
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset, DownloadConfig

# 1. 加载Flickr8k训练集
dataset = load_dataset(
    "intro/flickr8k",
    data_files={"train": "train/**"},
    split="train",
    download_config=DownloadConfig(max_retries=10)
)

# 2. 颜色词与候选替换颜色
colors = [
    "black", "white", "red", "blue",
    "green", "yellow", "pink", "purple",
    "orange", "brown", "gray", "grey"
]

# 只识别指定的颜色-物体组合
nouns = [
    "dress", "shirt", "shirts",
    "hat", "hats", "pants",
    "jacket", "coat", "uniform",
    "dog", "dogs", "cat", "cats",
    "car", "cars", "bus", "buses",
    "bicycle", "bike", "ball",
    "rope", "ropes", "fence",
    "skirt", "shoes", "helmet",
    "bag", "sign", "towels"
]

pattern = re.compile(
    r"\b(" + "|".join(colors) + r")\s+("
    + "|".join(nouns) + r")\b",
    re.IGNORECASE
)

output_dir = Path("hard_negative_candidates")
output_dir.mkdir(exist_ok=True)

records = []

# 3. 遍历6000张图片
for image_id in range(len(dataset)):
    row = dataset[image_id]

    captions = [
        row[f"caption_{j}"]
        for j in range(5)
    ]

    # 提取每条描述中的颜色+名词组合
    phrase_to_captions = {}

    for caption_id, caption in enumerate(captions):
        matches = {
            (m.group(1).lower(), m.group(2).lower())
            for m in pattern.finditer(caption)
        }

        for phrase in matches:
            phrase_to_captions.setdefault(
                phrase, []
            ).append(caption_id)

    # 优先选择至少两条描述共同提及的组合
    candidates = [
        (phrase, ids)
        for phrase, ids in phrase_to_captions.items()
        if len(ids) >= 2
    ]

    if not candidates:
        continue

    candidates.sort(
        key=lambda x: len(x[1]),
        reverse=True
    )

    (old_color, noun), caption_ids = candidates[0]

    positive = captions[caption_ids[0]]

    # 替换色不能出现在该图片的其他描述中
    observed_colors = {
        m.group(1).lower()
        for caption in captions
        for m in pattern.finditer(caption)
    }

    import random

    # 对动物使用更常见的毛色
    animal_nouns = {"dog", "dogs", "cat", "cats"}

    if noun in animal_nouns:
        allowed_colors = [
            "black", "white", "brown", "gray", "grey"
        ]
    else:
        allowed_colors = colors

    choices = [
        c for c in allowed_colors
        if c not in observed_colors
           and c != old_color
           and not (
                {c, old_color} == {"gray", "grey"}
        )
    ]

    replacement = (
        random.Random(image_id).choice(choices)
        if choices else None
    )

    if replacement is None:
        continue

    # 只替换目标颜色—名词组合一次
    replace_pattern = re.compile(
        rf"\b{re.escape(old_color)}\s+"
        rf"{re.escape(noun)}\b",
        re.IGNORECASE
    )

    negative = replace_pattern.sub(
        f"{replacement} {noun}",
        positive,
        count=1
    )
    # 修复替换颜色后可能出现的冠词错误
    negative = re.sub(
        r"\ban (?=(?:black|blue|brown|green|gray|grey|"
        r"pink|purple|red|white|yellow)\b)",
        "a ",
        negative,
        flags=re.IGNORECASE
    )

    negative = re.sub(
        r"\ba (?=(?:orange)\b)",
        "an ",
        negative,
        flags=re.IGNORECASE
    )

    if negative == positive:
        continue

    records.append({
        "image_id": image_id,
        "positive": positive,
        "negative": negative,
        "edit": f"{old_color} {noun} -> {replacement} {noun}",
        "status": "UNVERIFIED"
    })

# 4. 保存候选数据
csv_path = output_dir / "candidates.csv"

with open(
    csv_path, "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_id", "positive",
            "negative", "edit", "status"
        ]
    )
    writer.writeheader()
    writer.writerows(records)

# 5. 保存前12个候选对应的图片，供人工检查
preview_dir = output_dir / "preview"
preview_dir.mkdir(exist_ok=True)

for record in records[:12]:
    image_id = record["image_id"]
    dataset[image_id]["image"].save(
        preview_dir / f"image_{image_id}.jpg"
    )

# 6. 输出结果
print("\n=== Hard Negative Candidate Generation ===")
print("Training images:", len(dataset))
print("Candidates generated:", len(records))
print("CSV:", csv_path)

print("\n=== First 5 Candidates ===")

for record in records[:5]:
    print("\nImage ID:", record["image_id"])
    print("Positive:", record["positive"])
    print("Negative:", record["negative"])
    print("Edit:", record["edit"])
    print("Status:", record["status"])

print("\nCandidate generation finished.")
