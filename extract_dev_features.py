
import torch
import open_clip
from datasets import load_dataset, DownloadConfig

DEVICE = "cuda"
CAPTIONS_PER_IMAGE = 5

# 1. 加载公开测试集
print("Loading Flickr8k test dataset...")
dataset = load_dataset(
    "intro/flickr8k",
    data_files={"dev": "dev/**"},
    split="dev",
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

# 5. 检查训练特征
print("\n=== CLIP Training Features ===")
print("Train image features:", image_features.shape)
print("Train text features:", text_features.shape)

# 6. 保存训练特征
torch.save({
    "image_features": image_features,
    "text_features": text_features,
}, "clip_dev_features.pt")

print("\nTraining features saved.")