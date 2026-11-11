
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 基本设置
device = "cuda"
torch.manual_seed(42)

# 2. 加载之前保存的训练特征
data = torch.load(
    "clip_train_features.pt",
    map_location=device,
    weights_only=True
)

images = data["image_features"]
texts = data["text_features"]

num_images = len(images)

print("Images:", images.shape)
print("Texts:", texts.shape)


# 3. 定义一个小型Adapter神经网络
class Adapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 512)

        # 初始时不改变原始CLIP特征
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        update = self.fc2(F.relu(self.fc1(x)))
        return F.normalize(x + 0.2 * update, dim=-1)


# 4. 分别创建图像和文本Adapter
image_adapter = Adapter().to(device)
text_adapter = Adapter().to(device)

optimizer = torch.optim.AdamW(
    list(image_adapter.parameters()) +
    list(text_adapter.parameters()),
    lr=0.0005
)

initial_weight = image_adapter.fc2.weight.detach().clone()

# 5. 训练参数
epochs = 10
batch_size = 256
temperature = 0.07

print("\n=== Start E1 Adapter Training ===")

# 6. 开始训练
for epoch in range(epochs):
    image_adapter.train()
    text_adapter.train()

    total_loss = 0

    # 每轮随机打乱图片
    order = torch.randperm(num_images, device=device)

    for ids in order.split(batch_size):

        # 每张图片随机选择5条描述中的1条
        caption_ids = torch.randint(
            0, 5, (len(ids),), device=device
        )

        image_batch = images[ids]
        text_batch = texts[ids * 5 + caption_ids]

        # 前向传播
        image_output = image_adapter(image_batch)
        text_output = text_adapter(text_batch)

        # 计算图文相似度矩阵
        logits = image_output @ text_output.T
        logits = logits / temperature

        # batch内第i张图片对应第i条文字
        labels = torch.arange(len(ids), device=device)

        # 对称对比学习Loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2

        # 反向传播与参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(ids)

    avg_loss = total_loss / num_images

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Train Loss: {avg_loss:.4f}"
    )

# 7. 检查参数有没有真正发生变化
change = (
    image_adapter.fc2.weight - initial_weight
).abs().mean().item()

print("\nMean parameter change:", change)

# 8. 保存训练后的Adapter
torch.save({
    "image_adapter": image_adapter.state_dict(),
    "text_adapter": text_adapter.state_dict()
}, "clip_adapter_e1.pt")

print("Adapter saved successfully!")
