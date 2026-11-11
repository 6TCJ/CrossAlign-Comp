
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

    def forward(self, x):
        update = self.fc2(F.relu(self.fc1(x)))
        return F.normalize(x + 0.05 * update, dim=-1)


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


# 6. 输出E0与E1对比结果
print("\n=== E0 vs E1 on Flickr8k DEV ===")
print(f"{'Metric':<15} {'E0':>9} {'E1':>9} {'Change':>10}")

for direction in ["I2T", "T2I"]:
    for k in [1, 5, 10]:

        e0 = calculate_recall(
            e0_scores, k, direction
        )

        e1 = calculate_recall(
            e1_scores, k, direction
        )

        change = e1 - e0

        print(
            f"{direction + '@' + str(k):<15}"
            f"{e0:>8.2f}%"
            f"{e1:>8.2f}%"
            f"{change:>+9.2f}pp"
        )
