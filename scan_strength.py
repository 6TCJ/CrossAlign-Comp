
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 定义与训练时完全相同的Adapter结构
class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 512)

    def forward(self, x,strength = 0.2):
        update = self.fc2(F.relu(self.fc1(x)))
        return F.normalize(x + strength * update, dim=-1)


# 2. 加载Dev验证集特征
data = torch.load(
    "clip_dev_features.pt",
    map_location=device,
    weights_only=True
)

images = data["image_features"]
texts = data["text_features"]

print("Dev images:", images.shape)
print("Dev texts:", texts.shape)

assert len(texts) == len(images) * 5, "数据数量不匹配"

# 3. 加载训练好的Adapter参数
checkpoint = torch.load(
    "clip_adapter_e1.pt",
    map_location=device,
    weights_only=True
)

image_adapter = Adapter().to(device)
text_adapter = Adapter().to(device)

image_adapter.load_state_dict(
    checkpoint["image_adapter"]
)
text_adapter.load_state_dict(
    checkpoint["text_adapter"]
)

image_adapter.eval()
text_adapter.eval()

# 4. 分别计算E0和E1的相似度矩阵
with torch.inference_mode():

    # E0：完全不经过Adapter
    e0_scores = images @ texts.T

    # E1：经过训练后的Adapter
    new_images = image_adapter(images)
    new_texts = text_adapter(texts)

    e1_scores = new_images @ new_texts.T


# 5. 定义Recall@K评测函数
def calculate_recall(scores, k, direction):

    num_images = scores.shape[0]
    num_texts = scores.shape[1]

    if direction == "I2T":
        topk = scores.topk(k, dim=1).indices

        # 每张图片对应5条正确描述
        correct_ids = torch.arange(
            num_images, device=device
        ).unsqueeze(1)

        hits = (
            topk // 5 == correct_ids
        ).any(dim=1)

    else:
        topk = scores.T.topk(k, dim=1).indices

        # 每条文字只对应一张原始图片
        correct_ids = (
            torch.arange(num_texts, device=device) // 5
        ).unsqueeze(1)

        hits = (
            topk == correct_ids
        ).any(dim=1)

    return hits.float().mean().item() * 100


# 6. 测试不同的Adapter调整强度
print("\n=== Adapter Strength Ablation on DEV ===")
print(f"{'Strength':<12} {'I2T@1':>10} {'T2I@1':>10} {'T2I@5':>10}")

strengths = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30]

with torch.inference_mode():
    for strength in strengths:

        new_images = image_adapter(images, strength=strength)
        new_texts = text_adapter(texts, strength=strength)

        scores = new_images @ new_texts.T

        i2t_r1 = calculate_recall(scores, 1, "I2T")
        t2i_r1 = calculate_recall(scores, 1, "T2I")
        t2i_r5 = calculate_recall(scores, 5, "T2I")

        print(
            f"{strength:<12.3f}"
            f"{i2t_r1:>9.2f}%"
            f"{t2i_r1:>9.2f}%"
            f"{t2i_r5:>9.2f}%"
        )