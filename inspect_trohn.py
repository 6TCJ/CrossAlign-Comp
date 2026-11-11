
from datasets import load_dataset

# 流式读取，暂时不下载完整284MB数据
dataset = load_dataset(
    "imirandam/TROHN-Text",
    split="train",
    streaming=True
)

for i, row in enumerate(dataset):
    print(f"\n=== Sample {i + 1} ===")
    print("Image ID:", row["image_id"])
    print("Positive:", row["caption"])
    print("Negative:", row["negative_caption"])

    if i >= 9:
        break
