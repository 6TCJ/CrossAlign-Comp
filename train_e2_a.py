
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= 配置 =================
BASE = Path(__file__).resolve().parent
DATA = BASE / "e2_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
EPOCHS = 5
BATCH_SIZE = 128
HARD_BATCH = 8
LR = 2e-4

ALPHA = 0.05
TEMPERATURE = 0.07

HARD_MARGIN = 0.02
HARD_WEIGHT = 0.5

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ================= 1. Adapter结构 =================
class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def adapt(x, adapter):
    return F.normalize(
        x + ALPHA * adapter(x),
        dim=-1
    )


# ================= 2. 读取缓存 =================
def load_pt(path):
    return torch.load(
        path, map_location="cpu", weights_only=True
    )


train_cache = load_pt(BASE / "clip_train_features.pt")
dev_cache = load_pt(BASE / "clip_dev_features.pt")
hard_cache = load_pt(DATA / "e2_attr_features.pt")
checkpoint = load_pt(BASE / "clip_adapter_e1.pt")

train_images = F.normalize(
    train_cache["image_features"].float(), dim=-1
).to(DEVICE)

train_texts = F.normalize(
    train_cache["text_features"].float(), dim=-1
).to(DEVICE)

dev_images = F.normalize(
    dev_cache["image_features"].float(), dim=-1
).to(DEVICE)

dev_texts = F.normalize(
    dev_cache["text_features"].float(), dim=-1
).to(DEVICE)

hard_images = F.normalize(
    hard_cache["image_features"].float(), dim=-1
).to(DEVICE)

hard_pos = F.normalize(
    hard_cache["positive_features"].float(), dim=-1
).to(DEVICE)

hard_neg = F.normalize(
    hard_cache["negative_features"].float(), dim=-1
).to(DEVICE)

hard_train_idx = hard_cache["train_indices"].long()
hard_val_idx = hard_cache["val_indices"].long()

assert train_images.shape == (6000, 512)
assert train_texts.shape == (30000, 512)
assert dev_images.shape == (1000, 512)
assert dev_texts.shape == (5000, 512)
assert hard_images.shape[1] == 512

assert set(hard_train_idx.tolist()).isdisjoint(
    set(hard_val_idx.tolist())
)

print("Device:", DEVICE)
print("Flickr8k train images:", len(train_images))
print("Flickr8k train captions:", len(train_texts))
print("Hard train:", len(hard_train_idx))
print("Hard validation:", len(hard_val_idx))


# ================= 3. 恢复E1结构 =================
def build_adapters():
    image_state = checkpoint["image_adapter"]
    text_state = checkpoint["text_adapter"]

    image_hidden = image_state["fc1.weight"].shape[0]
    text_hidden = text_state["fc1.weight"].shape[0]

    image_adapter = Adapter(512, image_hidden).to(DEVICE)
    text_adapter = Adapter(512, text_hidden).to(DEVICE)

    image_adapter.load_state_dict(image_state, strict=True)
    text_adapter.load_state_dict(text_state, strict=True)

    return image_adapter, text_adapter


# ================= 4. 普通对比学习Loss =================
def contrastive_loss(image_f, text_f):
    logits = image_f @ text_f.T / TEMPERATURE

    labels = torch.arange(
        len(image_f), device=DEVICE
    )

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_i2t + loss_t2i)


# ================= 5. 困难负样本Loss =================
def hard_negative_loss(img_f, pos_f, neg_f):
    positive_score = (img_f * pos_f).sum(dim=-1)
    negative_score = (img_f * neg_f).sum(dim=-1)

    gap = positive_score - negative_score

    # 对正负分数差不足的样本施加惩罚
    return F.softplus(
        (HARD_MARGIN - gap) / TEMPERATURE
    ).mean()


# ================= 6. 评价函数 =================
@torch.no_grad()
def evaluate_retrieval(image_adapter, text_adapter):
    image_adapter.eval()
    text_adapter.eval()

    i = adapt(dev_images, image_adapter)
    t = adapt(dev_texts, text_adapter)

    sim = i @ t.T

    # 每张图片对应连续5条参考描述
    image_ids = torch.arange(1000, device=DEVICE)
    text_image_ids = torch.arange(5000, device=DEVICE) // 5

    i2t_r1 = (
        (sim.argmax(dim=1) // 5 == image_ids)
        .float().mean().item()
    )

    t2i_r1 = (
        (sim.argmax(dim=0) == text_image_ids)
        .float().mean().item()
    )

    return i2t_r1, t2i_r1


@torch.no_grad()
def evaluate_hard(image_adapter, text_adapter):
    image_adapter.eval()
    text_adapter.eval()

    idx = hard_val_idx.to(DEVICE)

    i = adapt(hard_images[idx], image_adapter)
    p = adapt(hard_pos[idx], text_adapter)
    n = adapt(hard_neg[idx], text_adapter)

    gaps = (i * p).sum(-1) - (i * n).sum(-1)

    accuracy = (gaps > 0).float().mean().item()
    mean_gap = gaps.mean().item()

    return accuracy, mean_gap


# ================= 7. 先复现E1 =================
e1_image_adapter, e1_text_adapter = build_adapters()

e1_i2t, e1_t2i = evaluate_retrieval(
    e1_image_adapter, e1_text_adapter
)

e1_hard_acc, e1_hard_gap = evaluate_hard(
    e1_image_adapter, e1_text_adapter
)

print("\n=== E1 Baseline Check ===")
print(f"Flickr8k I2T@1: {e1_i2t:.2%}")
print(f"Flickr8k T2I@1: {e1_t2i:.2%}")
print(f"Hard Val Accuracy: {e1_hard_acc:.2%}")
print(f"Hard Val Mean Gap: {e1_hard_gap:.5f}")

# 如果E1复现结果差异较大，停止训练并检查兼容性
if abs(e1_i2t - 0.8010) > 0.03 or \
   abs(e1_t2i - 0.6470) > 0.03:
    raise RuntimeError(
        "E1 Dev metrics do not match earlier results. "
        "Check feature ordering and Adapter implementation."
    )


# ================= 8. 训练函数 =================
def train_experiment(name, use_hard):
    print(f"\n=== Training {name} ===")

    # 两组实验均从同一E1权重出发
    image_adapter, text_adapter = build_adapters()

    optimizer = torch.optim.AdamW(
        list(image_adapter.parameters()) +
        list(text_adapter.parameters()),
        lr=LR
    )

    # 控制随机数，确保两组普通训练采样一致
    generator = torch.Generator().manual_seed(SEED)

    best_score = float("-inf")
    best_results = None

    save_path = DATA / f"{name}.pt"

    for epoch in range(EPOCHS):
        image_adapter.train()
        text_adapter.train()

        order = torch.randperm(
            len(train_images), generator=generator
        )

        # 每张图片每轮随机选择5条描述中的1条
        caption_choices = torch.randint(
            0, 5, (len(train_images),),
            generator=generator
        )

        total_loss = 0.0
        total_base = 0.0
        total_hard = 0.0
        steps = 0

        for start in range(0, len(order), BATCH_SIZE):
            image_ids = order[start:start + BATCH_SIZE]

            caption_ids = (
                image_ids * 5 +
                caption_choices[image_ids]
            )

            image_ids = image_ids.to(DEVICE)
            caption_ids = caption_ids.to(DEVICE)

            image_f = adapt(
                train_images[image_ids], image_adapter
            )
            text_f = adapt(
                train_texts[caption_ids], text_adapter
            )

            base_loss = contrastive_loss(
                image_f, text_f
            )

            # 为两组生成相同的困难样本索引
            sampled = torch.randint(
                0, len(hard_train_idx),
                (HARD_BATCH,),
                generator=generator
            )

            hard_ids = hard_train_idx[sampled].to(DEVICE)

            loss = base_loss
            h_loss_value = 0.0

            if use_hard:
                hi = adapt(
                    hard_images[hard_ids], image_adapter
                )
                hp = adapt(
                    hard_pos[hard_ids], text_adapter
                )
                hn = adapt(
                    hard_neg[hard_ids], text_adapter
                )

                h_loss = hard_negative_loss(hi, hp, hn)
                loss = base_loss + HARD_WEIGHT * h_loss
                h_loss_value = h_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_base += base_loss.item()
            total_hard += h_loss_value
            steps += 1

        i2t, t2i = evaluate_retrieval(
            image_adapter, text_adapter
        )

        hard_acc, hard_gap = evaluate_hard(
            image_adapter, text_adapter
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss={total_loss/steps:.4f} | "
            f"Base={total_base/steps:.4f} | "
            f"Hard={total_hard/steps:.4f} | "
            f"I2T@1={i2t:.2%} | "
            f"T2I@1={t2i:.2%} | "
            f"HardVal={hard_acc:.2%}"
        )

        # 两组统一按Flickr8k Dev检索指标选模型
        score = i2t + t2i

        if score > best_score:
            best_score = score

            best_results = {
                "epoch": epoch + 1,
                "i2t": i2t,
                "t2i": t2i,
                "hard_acc": hard_acc,
                "hard_gap": hard_gap
            }

            torch.save({
                "image_adapter": image_adapter.state_dict(),
                "text_adapter": text_adapter.state_dict(),
                "alpha": ALPHA,
                "epoch": epoch + 1,
                "use_hard": use_hard,
                "hard_weight": HARD_WEIGHT if use_hard else 0,
                "hard_margin": HARD_MARGIN,
                "temperature": TEMPERATURE
            }, save_path)

    print("Best epoch:", best_results["epoch"])
    print("Saved:", save_path)

    return best_results


# ================= 9. 两组公平对照 =================
control = train_experiment(
    "e2_a_control", use_hard=False
)

e2 = train_experiment(
    "e2_a_hard", use_hard=True
)


# ================= 10. 输出比较结果 =================
print("\n=== E1 vs Control vs E2-A ===")

print(
    f"{'Metric':<18}"
    f"{'E1':>12}"
    f"{'Control':>12}"
    f"{'E2-A':>12}"
)

print("-" * 54)

for label, key, e1_value in [
    ("Flickr I2T@1", "i2t", e1_i2t),
    ("Flickr T2I@1", "t2i", e1_t2i),
    ("Hard Val Acc", "hard_acc", e1_hard_acc)
]:
    print(
        f"{label:<18}"
        f"{e1_value:>11.2%} "
        f"{control[key]:>11.2%} "
        f"{e2[key]:>11.2%}"
    )

print("\nE2-A minus Control:")
print(
    "I2T@1:",
    f"{(e2['i2t']-control['i2t'])*100:+.2f} pp"
)
print(
    "T2I@1:",
    f"{(e2['t2i']-control['t2i'])*100:+.2f} pp"
)
print(
    "Hard Val:",
    f"{(e2['hard_acc']-control['hard_acc'])*100:+.2f} pp"
)

print("\nTraining finished.")
