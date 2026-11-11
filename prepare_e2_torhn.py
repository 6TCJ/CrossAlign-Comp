
import csv
import re
from pathlib import Path
from collections import Counter

from datasets import load_dataset

# 1. 创建E2数据文件夹
output_dir = Path("e2_data")
output_dir.mkdir(exist_ok=True)

print("Loading TROHN-Img training metadata...")

# 2. 流式读取，只选择文字与图片ID
# 这一阶段不下载negative_image图像内容
dataset = load_dataset(
    "imirandam/TROHN-Img",
    split="train",
    streaming=True,
    columns=[
        "image_id",
        "caption",
        "negative_caption"
    ]
)

# 3. 对文字进行简单分词
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

records = []
seen_images = set()

max_scan = 30000
max_keep = 3000

# 4. 筛选文本较相似的候选样本
for i, row in enumerate(dataset):

    if i >= max_scan or len(records) >= max_keep:
        break

    image_id = row["image_id"]
    positive = row["caption"].strip()
    negative = row["negative_caption"].strip()

    if image_id in seen_images:
        continue

    pos_tokens = tokenize(positive)
    neg_tokens = tokenize(negative)

    if not 6 <= len(pos_tokens) <= 35:
        continue

    if pos_tokens == neg_tokens:
        continue

    # 词汇重叠程度
    pos_counter = Counter(pos_tokens)
    neg_counter = Counter(neg_tokens)

    common = sum(
        (pos_counter & neg_counter).values()
    )

    overlap = common / max(
        len(pos_tokens), len(neg_tokens)
    )

    if overlap < 0.60:
        continue

    # 词汇相同但顺序不同的候选
    if pos_counter == neg_counter:
        candidate_type = "SWAP_LIKE"
    else:
        candidate_type = "HIGH_OVERLAP"

    records.append({
        "image_id": image_id,
        "positive": positive,
        "negative": negative,
        "candidate_type": candidate_type,
        "word_overlap": round(overlap, 4)
    })

    seen_images.add(image_id)

# 5. 保存候选元数据
csv_path = output_dir / "trohn_train_candidates.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_id",
            "positive",
            "negative",
            "candidate_type",
            "word_overlap"
        ]
    )

    writer.writeheader()
    writer.writerows(records)

# 6. 检查结果
swap_count = sum(
    r["candidate_type"] == "SWAP_LIKE"
    for r in records
)

print("\n=== TROHN-Img Preparation ===")
print("Candidates saved:", len(records))
print("Swap-like candidates:", swap_count)
print("CSV path:", csv_path)

print("\n=== First 5 Samples ===")

for row in records[:5]:
    print("\nImage:", row["image_id"])
    print("Positive:", row["positive"])
    print("Negative:", row["negative"])
    print("Type:", row["candidate_type"])
    print("Overlap:", row["word_overlap"])

print("\nPreparation finished.")
