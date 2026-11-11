
import json
import re
from pathlib import Path
from urllib.request import urlopen

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
strength = 0.05
batch_size = 16

# 1. 加载已经下载的SugarCrepe数据
dataset = load_dataset(
    "HuggingFaceM4/SugarCrepe_swap_att",
    split="test"
)
print("Samples:", len(dataset))

# 2. 获取官方原始答案
official_path = Path("swap_att_official.json")

if not official_path.exists():
    print("Downloading official annotations...")
    url = (
        "https://raw.githubusercontent.com/"
        "RAIVNLab/sugar-crepe/main/data/swap_att.json"
    )
    with urlopen(url, timeout=40) as response:
        official_path.write_bytes(response.read())

with open(official_path, "r", encoding="utf-8") as f:
    official = json.load(f)

def normalize_text(s):
    return re.sub(r"\s+", " ", s.strip())

# 将官方正确、错误描述映射成标准答案
answer_map = {}

for item in official.values():
    positive = normalize_text(item["caption"])
    negative = normalize_text(item["negative_caption"])
    key = tuple(sorted([positive, negative]))

    if key in answer_map:
        assert answer_map[key] == positive

    answer_map[key] = positive

# 3. 逐条检查标签是否与官方原始数据匹配
correct_indices = []

for i in range(len(dataset)):
    labels = dataset[i]["tested_labels"]
    assert len(labels) == 2

    normalized = [normalize_text(s) for s in labels]
    key = tuple(sorted(normalized))

    if key not in answer_map:
        raise ValueError(
            f"Sample {i} cannot match official annotations"
        )

    correct_index = normalized.index(answer_map[key])
    correct_indices.append(correct_index)

print("Verified labels:", len(correct_indices))

# 4. 定义与训练时相同的Adapter
class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 512)

    def forward(self, x):
        update = self.fc2(F.relu(self.fc1(x)))
        return F.normalize(x + strength * update, dim=-1)

checkpoint = torch.load(
    "clip_adapter_e1.pt",
    map_location="cpu",
    weights_only=True
)

image_adapter = Adapter().to(device)
text_adapter = Adapter().to(device)

image_adapter.load_state_dict(checkpoint["image_adapter"])
text_adapter.load_state_dict(checkpoint["text_adapter"])

image_adapter.eval()
text_adapter.eval()

# 5. 加载与E0完全相同的OpenCLIP
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

e0_correct = 0
e1_correct = 0
e1_fixes = []   # E0做错、E1做对的题
e1_breaks = []  # E0做对、E1做错的题

# 6. 分批进行图文匹配评测
with torch.inference_mode():
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        rows = [dataset[i] for i in range(start, end)]
        n = len(rows)

        image_batch = torch.stack([
            preprocess(row["image"].convert("RGB"))
            for row in rows
        ]).to(device)

        captions = [
            caption
            for row in rows
            for caption in row["tested_labels"]
        ]
        text_batch = tokenizer(captions).to(device)

        # 原始CLIP特征
        image_features = F.normalize(
            model.encode_image(image_batch), dim=-1
        )
        text_features = F.normalize(
            model.encode_text(text_batch), dim=-1
        )

        # 一张图片对应两条文字
        text_pairs = text_features.reshape(n, 2, 512)

        # E0：原始CLIP
        scores_e0 = (
            image_features.unsqueeze(1) * text_pairs
        ).sum(dim=-1)

        # E1：经过Adapter调整的特征
        new_images = image_adapter(image_features)
        new_texts = text_adapter(text_features)
        new_texts = new_texts.reshape(n, 2, 512)

        scores_e1 = (
            new_images.unsqueeze(1) * new_texts
        ).sum(dim=-1)

        targets = torch.tensor(
            correct_indices[start:end],
            device=device
        )

        e0_correct += (
            scores_e0.argmax(dim=1) == targets
        ).sum().item()

        e1_correct += (
            scores_e1.argmax(dim=1) == targets
        ).sum().item()


        # 记录E0与E1预测结果不一致的样本
        pred0 = scores_e0.argmax(dim=1).cpu().tolist()
        pred1 = scores_e1.argmax(dim=1).cpu().tolist()

        for j, (p0, p1) in enumerate(zip(pred0, pred1)):
            sample_id = start + j
            correct = correct_indices[sample_id]

            if p0 != correct and p1 == correct:
                e1_fixes.append(sample_id)

            elif p0 == correct and p1 != correct:
                e1_breaks.append(sample_id)

        print(f"Evaluated: {end}/{len(dataset)}")

# 7. 输出结果
total = len(dataset)

e0_accuracy = e0_correct / total * 100
e1_accuracy = e1_correct / total * 100

print("\n=== SugarCrepe Swap-Att Results ===")
print(f"Total samples: {total}")
print(f"E0 correct: {e0_correct}")
print(f"E1 correct: {e1_correct}")
print(f"E0 Accuracy: {e0_accuracy:.2f}%")
print(f"E1 Accuracy: {e1_accuracy:.2f}%")
print(f"Change: {e1_accuracy - e0_accuracy:+.2f}pp")

print("\n=== Error Analysis ===")
print("E1 fixed errors:", len(e1_fixes))
print("E1 introduced errors:", len(e1_breaks))
print("Net improvement:", len(e1_fixes) - len(e1_breaks))

print("\n=== Examples fixed by E1 ===")
print("\n=== Examples broken by E1 ===")

for i in e1_breaks[:5]:
    labels = dataset[i]["tested_labels"]
    correct = correct_indices[i]

    print(f"\nSample {i}")
    print("Correct:", labels[correct])
    print("Wrong:  ", labels[1 - correct])
for i in e1_fixes[:3]:
    labels = dataset[i]["tested_labels"]
    correct = correct_indices[i]

    print(f"\nSample {i}")
    print("Correct:", labels[correct])
    print("Wrong:  ", labels[1 - correct])