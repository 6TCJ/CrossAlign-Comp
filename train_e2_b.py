"""
CrossAlign E2-B: Adapter training with TROHN-Img hard-negative image/text pairs.

The CLIP image/text encoders remain frozen. Only the existing E1 image and text
Adapters are updated. Each TROHN item contributes:
  original COCO image <-> positive caption
  generated negative image <-> negative caption

The TROHN loss is the same symmetric image-text contrastive objective used by
the official BiVLC implementation, adapted to CrossAlign's frozen-encoder +
Adapter setting. Flickr8k base loss is retained for continuity with E2-A.
"""

import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "e2_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
EPOCHS = 5
BATCH_SIZE = 128
TROHN_BATCH = 8
LR = 2e-4

ALPHA = 0.05
TEMPERATURE = 0.07
TROHN_WEIGHT = 0.5

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_pt(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def adapt(x, adapter):
    return F.normalize(x + ALPHA * adapter(x), dim=-1)


def build_adapters(checkpoint):
    image_state = checkpoint["image_adapter"]
    text_state = checkpoint["text_adapter"]

    image_adapter = Adapter(
        512, image_state["fc1.weight"].shape[0]
    ).to(DEVICE)
    text_adapter = Adapter(
        512, text_state["fc1.weight"].shape[0]
    ).to(DEVICE)

    image_adapter.load_state_dict(image_state, strict=True)
    text_adapter.load_state_dict(text_state, strict=True)
    return image_adapter, text_adapter


def contrastive_loss(image_f, text_f):
    logits = image_f @ text_f.T / TEMPERATURE
    labels = torch.arange(len(image_f), device=DEVICE)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)


# Flickr8k caches and E1 checkpoint.
train_cache = load_pt(ROOT / "clip_train_features.pt")
dev_cache = load_pt(ROOT / "clip_dev_features.pt")
trohn_cache = load_pt(DATA / "e2_b_trohn_features.pt")
e1_checkpoint = load_pt(ROOT / "clip_adapter_e1.pt")

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

trohn_images = F.normalize(
    trohn_cache["image_features"].float(), dim=-1
).to(DEVICE)
trohn_negative_images = F.normalize(
    trohn_cache["negative_image_features"].float(), dim=-1
).to(DEVICE)
trohn_positive = F.normalize(
    trohn_cache["positive_features"].float(), dim=-1
).to(DEVICE)
trohn_negative = F.normalize(
    trohn_cache["negative_features"].float(), dim=-1
).to(DEVICE)

trohn_train_idx = trohn_cache["train_indices"].long()
trohn_val_idx = trohn_cache["val_indices"].long()

assert train_images.shape == (6000, 512), train_images.shape
assert train_texts.shape == (30000, 512), train_texts.shape
assert dev_images.shape == (1000, 512), dev_images.shape
assert dev_texts.shape == (5000, 512), dev_texts.shape
assert trohn_images.shape == trohn_negative_images.shape
assert trohn_images.shape == trohn_positive.shape == trohn_negative.shape
assert trohn_images.shape[1] == 512
assert len(trohn_train_idx) > 0 and len(trohn_val_idx) > 0
assert set(trohn_train_idx.tolist()).isdisjoint(trohn_val_idx.tolist())

print("Device:", DEVICE)
print("Flickr8k train images:", len(train_images))
print("TROHN-Img pilot pairs:", len(trohn_images))
print("TROHN train pairs:", len(trohn_train_idx))
print("TROHN validation pairs:", len(trohn_val_idx))


@torch.no_grad()
def evaluate_retrieval(image_adapter, text_adapter):
    image_adapter.eval()
    text_adapter.eval()

    image_f = adapt(dev_images, image_adapter)
    text_f = adapt(dev_texts, text_adapter)
    similarity = image_f @ text_f.T

    image_ids = torch.arange(len(dev_images), device=DEVICE)
    text_image_ids = torch.arange(len(dev_texts), device=DEVICE) // 5

    i2t = (
        (similarity.argmax(dim=1) // 5 == image_ids)
        .float().mean().item()
    )
    t2i = (
        (similarity.argmax(dim=0) == text_image_ids)
        .float().mean().item()
    )
    return i2t, t2i


@torch.no_grad()
def evaluate_trohn(image_adapter, text_adapter):
    image_adapter.eval()
    text_adapter.eval()

    idx = trohn_val_idx.to(DEVICE)
    i0 = adapt(trohn_images[idx], image_adapter)
    i1 = adapt(trohn_negative_images[idx], image_adapter)
    t0 = adapt(trohn_positive[idx], text_adapter)
    t1 = adapt(trohn_negative[idx], text_adapter)

    s00 = (i0 * t0).sum(dim=-1)
    s01 = (i0 * t1).sum(dim=-1)
    s10 = (i1 * t0).sum(dim=-1)
    s11 = (i1 * t1).sum(dim=-1)

    # Both directions must choose the corresponding positive pair.
    i2t = ((s00 > s01) & (s11 > s10)).float().mean().item()
    t2i = ((s00 > s10) & (s11 > s01)).float().mean().item()
    group = (
        (s00 > s01) & (s11 > s10) &
        (s00 > s10) & (s11 > s01)
    ).float().mean().item()

    return i2t, t2i, group


def train_e2b():
    print("\n=== E1 baseline check ===")
    e1_image_adapter, e1_text_adapter = build_adapters(e1_checkpoint)
    e1_i2t, e1_t2i = evaluate_retrieval(
        e1_image_adapter, e1_text_adapter
    )
    print(f"E1 Flickr I2T@1: {e1_i2t:.2%}")
    print(f"E1 Flickr T2I@1: {e1_t2i:.2%}")

    image_adapter, text_adapter = build_adapters(e1_checkpoint)
    optimizer = torch.optim.AdamW(
        list(image_adapter.parameters()) +
        list(text_adapter.parameters()),
        lr=LR,
    )

    generator = torch.Generator().manual_seed(SEED)
    save_path = DATA / "e2_b_trohn_img.pt"
    best_score = float("-inf")
    best_result = None

    print("\n=== Training E2-B TROHN-Img Adapter ===")
    for epoch in range(EPOCHS):
        image_adapter.train()
        text_adapter.train()

        order = torch.randperm(len(train_images), generator=generator)
        caption_choices = torch.randint(
            0, 5, (len(train_images),), generator=generator
        )

        total_loss = 0.0
        total_base = 0.0
        total_trohn = 0.0
        steps = 0

        for start in range(0, len(order), BATCH_SIZE):
            image_ids = order[start:start + BATCH_SIZE]
            caption_ids = image_ids * 5 + caption_choices[image_ids]

            image_ids_device = image_ids.to(DEVICE)
            caption_ids_device = caption_ids.to(DEVICE)

            base_image_f = adapt(
                train_images[image_ids_device], image_adapter
            )
            base_text_f = adapt(
                train_texts[caption_ids_device], text_adapter
            )
            base_loss = contrastive_loss(base_image_f, base_text_f)

            sampled = torch.randint(
                0, len(trohn_train_idx), (TROHN_BATCH,),
                generator=generator
            )
            trohn_ids = trohn_train_idx[sampled].to(DEVICE)

            # Two aligned pairs per TROHN row:
            # [original image, generated negative image]
            # [positive caption, negative caption]
            trohn_image_f = torch.cat([
                adapt(trohn_images[trohn_ids], image_adapter),
                adapt(trohn_negative_images[trohn_ids], image_adapter),
            ], dim=0)
            trohn_text_f = torch.cat([
                adapt(trohn_positive[trohn_ids], text_adapter),
                adapt(trohn_negative[trohn_ids], text_adapter),
            ], dim=0)

            trohn_loss = contrastive_loss(
                trohn_image_f, trohn_text_f
            )
            loss = base_loss + TROHN_WEIGHT * trohn_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_base += base_loss.item()
            total_trohn += trohn_loss.item()
            steps += 1

        i2t, t2i = evaluate_retrieval(image_adapter, text_adapter)
        trohn_i2t, trohn_t2i, trohn_group = evaluate_trohn(
            image_adapter, text_adapter
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss={total_loss / steps:.4f} | "
            f"Base={total_base / steps:.4f} | "
            f"TROHN={total_trohn / steps:.4f} | "
            f"Flickr I2T@1={i2t:.2%} | "
            f"Flickr T2I@1={t2i:.2%} | "
            f"TROHN I2T={trohn_i2t:.2%} | "
            f"TROHN T2I={trohn_t2i:.2%} | "
            f"Group={trohn_group:.2%}"
        )

        # Keep the same checkpoint-selection rule as E2-A for comparability.
        score = i2t + t2i
        if score > best_score:
            best_score = score
            best_result = {
                "epoch": epoch + 1,
                "flickr_i2t": i2t,
                "flickr_t2i": t2i,
                "trohn_i2t": trohn_i2t,
                "trohn_t2i": trohn_t2i,
                "trohn_group": trohn_group,
            }
            torch.save({
                "image_adapter": image_adapter.state_dict(),
                "text_adapter": text_adapter.state_dict(),
                "alpha": ALPHA,
                "epoch": epoch + 1,
                "use_trohn_img": True,
                "trohn_weight": TROHN_WEIGHT,
                "trohn_batch": TROHN_BATCH,
                "temperature": TEMPERATURE,
                "selection_metric": "Flickr8k I2T@1 + T2I@1",
            }, save_path)

    print("\nBest E2-B epoch:", best_result["epoch"])
    print("Saved:", save_path)
    print("Best result:", best_result)


@torch.no_grad()
def evaluate_checkpoint(path):
    checkpoint = load_pt(path)
    image_adapter, text_adapter = build_adapters(checkpoint)
    i2t, t2i = evaluate_retrieval(image_adapter, text_adapter)
    trohn_i2t, trohn_t2i, group = evaluate_trohn(
        image_adapter, text_adapter
    )
    return {
        "model": path.stem,
        "Flickr_I2T@1": i2t,
        "Flickr_T2I@1": t2i,
        "TROHN_I2T": trohn_i2t,
        "TROHN_T2I": trohn_t2i,
        "TROHN_Group": group,
    }



@torch.no_grad()
def compare_models():

    candidates = [
        ("E1", ROOT / "clip_adapter_e1.pt"),
        ("E2-A-Control", DATA / "e2_a_control.pt"),
        ("E2-A-Hard", DATA / "e2_a_hard.pt"),
        ("E2-B-Full", DATA / "e2_b_trohn_img_full_best.pt"),
    ]

    # 检查四个模型权重
    missing = [str(p) for _, p in candidates if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model checkpoints:\n" + "\n".join(missing)
        )

    # 使用完整 TROHN 验证集，而不是373条pilot数据
    val_dir = DATA / "e2_b_full" / "validation"
    shards = sorted(val_dir.glob("shard_*.pt"))

    if not shards:
        raise RuntimeError("Full TROHN validation shards not found.")

    expected_rows = 59214
    batch_size = 1024

    print("\nFull TROHN validation shards:", len(shards))

    def evaluate_full_trohn(image_adapter, text_adapter):

        total = 0
        correct_i2t = 0
        correct_t2i = 0
        correct_group = 0

        for shard_path in shards:

            data = load_pt(shard_path)
            n = len(data["image_features"])

            for start in range(0, n, batch_size):

                end = min(start + batch_size, n)

                i0 = F.normalize(
                    data["image_features"][start:end].float(),
                    dim=-1
                ).to(DEVICE)

                i1 = F.normalize(
                    data["negative_image_features"][start:end].float(),
                    dim=-1
                ).to(DEVICE)

                t0 = F.normalize(
                    data["positive_features"][start:end].float(),
                    dim=-1
                ).to(DEVICE)

                t1 = F.normalize(
                    data["negative_features"][start:end].float(),
                    dim=-1
                ).to(DEVICE)

                i0 = adapt(i0, image_adapter)
                i1 = adapt(i1, image_adapter)
                t0 = adapt(t0, text_adapter)
                t1 = adapt(t1, text_adapter)

                s00 = (i0 * t0).sum(-1)
                s01 = (i0 * t1).sum(-1)
                s10 = (i1 * t0).sum(-1)
                s11 = (i1 * t1).sum(-1)

                ok_i2t = (s00 > s01) & (s11 > s10)
                ok_t2i = (s00 > s10) & (s11 > s01)
                ok_group = ok_i2t & ok_t2i

                correct_i2t += ok_i2t.sum().item()
                correct_t2i += ok_t2i.sum().item()
                correct_group += ok_group.sum().item()

                total += end - start

            print(
                f"TROHN validation: {total}/{expected_rows}",
                flush=True
            )

        if total != expected_rows:
            raise RuntimeError(
                f"Validation row mismatch: {total}/{expected_rows}"
            )

        return (
            correct_i2t / total,
            correct_t2i / total,
            correct_group / total
        )

    results = []

    print("\n========== FINAL FOUR-MODEL COMPARISON ==========")

    for label, path in candidates:

        print(f"\nEvaluating: {label}")

        checkpoint = load_pt(path)

        image_adapter, text_adapter = build_adapters(checkpoint)

        # Flickr8k Dev
        flickr_i2t, flickr_t2i = evaluate_retrieval(
            image_adapter, text_adapter
        )

        # Full TROHN-Img validation
        trohn_i2t, trohn_t2i, trohn_group = evaluate_full_trohn(
            image_adapter, text_adapter
        )

        row = {
            "model": label,
            "Flickr_I2T@1": flickr_i2t,
            "Flickr_T2I@1": flickr_t2i,
            "TROHN_I2T": trohn_i2t,
            "TROHN_T2I": trohn_t2i,
            "TROHN_Group": trohn_group,
        }

        results.append(row)

        print(
            f"{label:18s} | "
            f"Flickr {flickr_i2t:.2%}/{flickr_t2i:.2%} | "
            f"TROHN {trohn_i2t:.2%}/{trohn_t2i:.2%} | "
            f"Group {trohn_group:.2%}"
        )

        del image_adapter, text_adapter, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存最终比较结果
    output = DATA / "e2_all_models_full_comparison.csv"

    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n========== COMPARISON COMPLETE ==========")
    print("Models evaluated:", len(results))
    print("TROHN validation rows:", expected_rows)
    print("Results saved:", output)


if __name__ == "__main__":
    print("\n=== FINAL E1 / E2-A-Control / E2-A-Hard / E2-B COMPARISON ===")
    compare_models()
