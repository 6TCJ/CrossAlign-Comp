
import torch
import open_clip
from datasets import load_dataset, DownloadConfig

DEVICE = "cuda"
CAPTIONS_PER_IMAGE = 5

# 1. 加载公开测试集
print("Loading Flickr8k test dataset...")
dataset = load_dataset(
    "intro/flickr8k",
    data_files={"test": "test/**"},
    split="test",
    download_config=DownloadConfig(max_retries=10)
)
print("Test images:", len(dataset))
print("Columns:", dataset.column_names)

# 2. 加载与第一次实验完全相同的CLIP
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image_features = []
all_captions = []

# 3. 提取全部图片Embedding
with torch.inference_mode():
    for start in range(0, len(dataset), 32):
        rows = [
            dataset[i]
            for i in range(start, min(start + 32, len(dataset)))
        ]

        images = torch.stack([
            preprocess(row["image"].convert("RGB"))
            for row in rows
        ]).to(DEVICE)

        features = model.encode_image(images)
        features = torch.nn.functional.normalize(features, dim=-1)
        image_features.append(features.cpu())

        for row in rows:
            for j in range(CAPTIONS_PER_IMAGE):
                all_captions.append(row[f"caption_{j}"])

        print(f"Images: {min(start + 32, len(dataset))}/{len(dataset)}")

# 4. 提取全部文字Embedding
text_features = []

with torch.inference_mode():
    for start in range(0, len(all_captions), 256):
        captions = all_captions[start:start + 256]
        tokens = tokenizer(captions).to(DEVICE)

        features = model.encode_text(tokens)
        features = torch.nn.functional.normalize(features, dim=-1)
        text_features.append(features.cpu())

image_features = torch.cat(image_features).float()
text_features = torch.cat(text_features).float()

# 5. 图文相似度矩阵
scores = image_features @ text_features.T

# 6. 图像找文字：前K条中是否有任意正确描述
def image_to_text_recall(k):
    topk = scores.topk(k, dim=1).indices
    correct_image_ids = torch.arange(len(dataset)).unsqueeze(1)
    hits = (topk // CAPTIONS_PER_IMAGE == correct_image_ids).any(dim=1)
    return hits.float().mean().item()

# 7. 文字找图像：前K张中是否包含正确图片
def text_to_image_recall(k):
    topk = scores.T.topk(k, dim=1).indices
    correct_image_ids = (
        torch.arange(len(all_captions)) // CAPTIONS_PER_IMAGE
    ).unsqueeze(1)
    hits = (topk == correct_image_ids).any(dim=1)
    return hits.float().mean().item()

print("\n=== E0 Original CLIP Baseline ===")
print("Image features:", image_features.shape)
print("Text features:", text_features.shape)

for k in [1, 5, 10]:
    print(
        f"Recall@{k}: "
        f"I2T={image_to_text_recall(k):.4f}, "
        f"T2I={text_to_image_recall(k):.4f}"
    )

# 保存基线特征，后续用于实验对照
torch.save({
    "image_features": image_features,
    "text_features": text_features,
}, "clip_baseline_features.pt")

print("\nBaseline features saved.")
