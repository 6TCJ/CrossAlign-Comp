
import csv
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= 配置 =================
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "e2_data"

TRAIN_DIR = DATA / "e2_b_full" / "train"
VAL_DIR = DATA / "e2_b_full" / "validation"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
EPOCHS = 5
BATCH_SIZE = 128
TROHN_BATCH = 8

LR = 2e-4
ALPHA = 0.05
TEMPERATURE = 0.07
TROHN_WEIGHT = 0.5

FEATURE_KEYS = [
    "image_features",
    "negative_image_features",
    "positive_features",
    "negative_features",
]

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_pt(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(
        path, map_location="cpu", weights_only=True
    )


# ================= Adapter =================
class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def adapt(x, adapter):
    return F.normalize(
        x + ALPHA * adapter(x), dim=-1
    )


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


# ================= Flickr 对比损失 =================
def contrastive_loss(image_f, text_f):
    logits = image_f @ text_f.T / TEMPERATURE

    labels = torch.arange(
        len(image_f), device=DEVICE
    )

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_i2t + loss_t2i)


# ================= TROHN 对比损失 =================
def trohn_group_loss(batch, image_adapter, text_adapter):

    n = batch["image_features"].shape[0]

    assert n % TROHN_BATCH == 0, n

    # 原始图像、负样本图像
    i0 = adapt(
        F.normalize(batch["image_features"], dim=-1),
        image_adapter
    )

    i1 = adapt(
        F.normalize(batch["negative_image_features"], dim=-1),
        image_adapter
    )

    # 正向文本、负向文本
    t0 = adapt(
        F.normalize(batch["positive_features"], dim=-1),
        text_adapter
    )

    t1 = adapt(
        F.normalize(batch["negative_features"], dim=-1),
        text_adapter
    )

    # 每条记录构成两个匹配对：
    # 原图 <-> 正文本
    # 负图 <-> 负文本

    image_pairs = torch.stack(
        [i0, i1], dim=1
    ).reshape(-1, 512)

    text_pairs = torch.stack(
        [t0, t1], dim=1
    ).reshape(-1, 512)

    groups = n // TROHN_BATCH

    image_pairs = image_pairs.reshape(
        groups, TROHN_BATCH * 2, 512
    )

    text_pairs = text_pairs.reshape(
        groups, TROHN_BATCH * 2, 512
    )

    # 每8条原始记录构成一个16x16对比矩阵
    logits = torch.bmm(
        image_pairs,
        text_pairs.transpose(1, 2)
    ) / TEMPERATURE

    labels = torch.arange(
        TROHN_BATCH * 2, device=DEVICE
    ).repeat(groups)

    loss_i2t = F.cross_entropy(
        logits.reshape(-1, TROHN_BATCH * 2),
        labels
    )

    loss_t2i = F.cross_entropy(
        logits.transpose(1, 2).reshape(
            -1, TROHN_BATCH * 2
        ),
        labels
    )

    return 0.5 * (loss_i2t + loss_t2i)


# ================= 检查全量数据 =================
def inspect_shards(paths, name):

    total = 0

    for p in paths:
        d = load_pt(p)

        n = d["image_features"].shape[0]

        for key in FEATURE_KEYS:
            x = d[key]

            if x.shape != (n, 512):
                raise RuntimeError(
                    f"Bad feature shape: {p} {key} {x.shape}"
                )

        total += n
        del d

    print(
        f"{name}: {len(paths)} shards, "
        f"{total:,} rows"
    )

    return total


# ================= 分片读取与批量组合 =================
def iter_trohn_batches(paths, rows_per_update, epoch):

    files = list(paths)
    random.Random(SEED + epoch).shuffle(files)

    generator = torch.Generator().manual_seed(
        SEED + epoch
    )

    buffers = {k: [] for k in FEATURE_KEYS}
    buffered = 0

    for file_idx, path in enumerate(files, 1):

        d = load_pt(path)
        n = len(d["image_features"])

        order = torch.randperm(
            n, generator=generator
        )

        pos = 0

        while pos < n:

            take = min(
                rows_per_update - buffered,
                n - pos
            )

            ids = order[pos:pos + take]

            for key in FEATURE_KEYS:
                buffers[key].append(
                    d[key][ids].float()
                )

            buffered += take
            pos += take

            if buffered == rows_per_update:

                result = {
                    key: torch.cat(
                        buffers[key], dim=0
                    ).to(DEVICE)
                    for key in FEATURE_KEYS
                }

                yield result

                buffers = {k: [] for k in FEATURE_KEYS}
                buffered = 0

        del d

        if file_idx % 10 == 0:
            print(
                f"Loaded train shards: "
                f"{file_idx}/{len(files)}",
                flush=True
            )

    if buffered > 0:

        result = {
            key: torch.cat(
                buffers[key], dim=0
            ).to(DEVICE)
            for key in FEATURE_KEYS
        }

        yield result


# ================= Flickr 验证 =================
@torch.no_grad()
def evaluate_retrieval(image_adapter, text_adapter):

    image_adapter.eval()
    text_adapter.eval()

    image_f = adapt(dev_images, image_adapter)
    text_f = adapt(dev_texts, text_adapter)

    similarity = image_f @ text_f.T

    image_ids = torch.arange(
        len(dev_images), device=DEVICE
    )

    text_image_ids = torch.arange(
        len(dev_texts), device=DEVICE
    ) // 5

    i2t = (
        (similarity.argmax(dim=1) // 5 == image_ids)
        .float().mean().item()
    )

    t2i = (
        (similarity.argmax(dim=0) == text_image_ids)
        .float().mean().item()
    )

    return i2t, t2i


# ================= 全量 TROHN 验证 =================
@torch.no_grad()
def evaluate_trohn(image_adapter, text_adapter):

    image_adapter.eval()
    text_adapter.eval()

    i2t_correct = 0
    t2i_correct = 0
    group_correct = 0
    total = 0

    for file_idx, path in enumerate(val_shards, 1):

        d = load_pt(path)

        n = len(d["image_features"])

        for start in range(0, n, 1024):

            end = min(start + 1024, n)

            i0 = adapt(
                F.normalize(
                    d["image_features"][start:end].float().to(DEVICE),
                    dim=-1
                ),
                image_adapter
            )

            i1 = adapt(
                F.normalize(
                    d["negative_image_features"][start:end].float().to(DEVICE),
                    dim=-1
                ),
                image_adapter
            )

            t0 = adapt(
                F.normalize(
                    d["positive_features"][start:end].float().to(DEVICE),
                    dim=-1
                ),
                text_adapter
            )

            t1 = adapt(
                F.normalize(
                    d["negative_features"][start:end].float().to(DEVICE),
                    dim=-1
                ),
                text_adapter
            )

            s00 = (i0 * t0).sum(dim=-1)
            s01 = (i0 * t1).sum(dim=-1)
            s10 = (i1 * t0).sum(dim=-1)
            s11 = (i1 * t1).sum(dim=-1)

            a = (s00 > s01) & (s11 > s10)
            b = (s00 > s10) & (s11 > s01)

            i2t_correct += a.sum().item()
            t2i_correct += b.sum().item()
            group_correct += (a & b).sum().item()

            total += end - start

        del d

        print(
            f"Validation shards: "
            f"{file_idx}/{len(val_shards)}",
            flush=True
        )

    return (
        i2t_correct / total,
        t2i_correct / total,
        group_correct / total
    )


# ================= 主训练 =================
def train():

    global dev_images, dev_texts
    global val_shards

    if DEVICE != "cuda":
        raise RuntimeError("CUDA GPU not detected!")

    print("GPU:", torch.cuda.get_device_name(0))

    train_shards = sorted(TRAIN_DIR.glob("shard_*.pt"))
    val_shards = sorted(VAL_DIR.glob("shard_*.pt"))

    if not train_shards or not val_shards:
        raise RuntimeError("TROHN shard files missing!")

    print("\n=== Data preflight ===")

    train_rows = inspect_shards(train_shards, "Train")
    val_rows = inspect_shards(val_shards, "Validation")

    if train_rows != 236856 or val_rows != 59214:
        raise RuntimeError(
            f"Unexpected row counts: {train_rows}, {val_rows}"
        )

    # Flickr8k features
    train_cache = load_pt(ROOT / "clip_train_features.pt")
    dev_cache = load_pt(ROOT / "clip_dev_features.pt")

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

    assert train_images.shape == (6000, 512)
    assert train_texts.shape == (30000, 512)
    assert dev_images.shape == (1000, 512)
    assert dev_texts.shape == (5000, 512)

    # E1 Adapter initialization
    e1_checkpoint = load_pt(
        ROOT / "clip_adapter_e1.pt"
    )

    image_adapter, text_adapter = build_adapters(
        e1_checkpoint
    )

    optimizer = torch.optim.AdamW(
        list(image_adapter.parameters()) +
        list(text_adapter.parameters()),
        lr=LR
    )

    # Keep optimizer updates aligned with Flickr batches.
    base_steps = math.ceil(len(train_images) / BATCH_SIZE)

    rows_per_update = (
        math.ceil(
            train_rows / base_steps / TROHN_BATCH
        ) * TROHN_BATCH
    )

    trohn_updates = math.ceil(
        train_rows / rows_per_update
    )

    assert trohn_updates == base_steps

    print("\n=== Training configuration ===")
    print("Train rows:", train_rows)
    print("Validation rows:", val_rows)
    print("Epochs:", EPOCHS)
    print("Flickr batch:", BATCH_SIZE)
    print("TROHN contrastive group:", TROHN_BATCH)
    print("Rows per optimizer update:", rows_per_update)
    print("Optimizer updates per epoch:", base_steps)
    print("Learning rate:", LR)

    # E1 baseline
    print("\n=== E1 baseline ===")

    e1_i2t, e1_t2i = evaluate_retrieval(
        image_adapter, text_adapter
    )

    e1_trohn = evaluate_trohn(
        image_adapter, text_adapter
    )

    print(
        f"E1 Flickr: {e1_i2t:.2%}/{e1_t2i:.2%}"
    )

    print(
        f"E1 TROHN: "
        f"I2T={e1_trohn[0]:.2%} "
        f"T2I={e1_trohn[1]:.2%} "
        f"Group={e1_trohn[2]:.2%}"
    )

    best_score = float("-inf")

    best_path = DATA / "e2_b_trohn_img_full_best.pt"
    latest_path = DATA / "e2_b_trohn_img_full_latest.pt"
    history_path = DATA / "e2_b_trohn_img_full_history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "loss", "base_loss", "trohn_loss",
            "flickr_i2t", "flickr_t2i",
            "trohn_i2t", "trohn_t2i", "trohn_group"
        ])

    # ================= Epoch loop =================
    for epoch in range(1, EPOCHS + 1):

        print(f"\n========== EPOCH {epoch}/{EPOCHS} ==========")

        image_adapter.train()
        text_adapter.train()

        generator = torch.Generator().manual_seed(
            SEED + 1000 + epoch
        )

        base_order = torch.randperm(
            len(train_images), generator=generator
        )

        caption_choices = torch.randint(
            0, 5, (len(train_images),),
            generator=generator
        )

        trohn_iter = iter(
            iter_trohn_batches(
                train_shards,
                rows_per_update,
                epoch
            )
        )

        total_loss = 0.0
        total_base = 0.0
        total_trohn = 0.0

        for step in range(base_steps):

            try:
                trohn_batch = next(trohn_iter)
            except StopIteration:
                raise RuntimeError(
                    "TROHN data ended before epoch completion!"
                )

            start = step * BATCH_SIZE

            image_ids = base_order[
                start:start + BATCH_SIZE
            ]

            caption_ids = (
                image_ids * 5 + caption_choices[image_ids]
            )

            image_ids = image_ids.to(DEVICE)
            caption_ids = caption_ids.to(DEVICE)

            base_image_f = adapt(
                train_images[image_ids], image_adapter
            )

            base_text_f = adapt(
                train_texts[caption_ids], text_adapter
            )

            base_loss = contrastive_loss(
                base_image_f, base_text_f
            )

            trohn_loss = trohn_group_loss(
                trohn_batch,
                image_adapter,
                text_adapter
            )

            loss = base_loss + TROHN_WEIGHT * trohn_loss

            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_base += base_loss.item()
            total_trohn += trohn_loss.item()

            del trohn_batch

            if (
                (step + 1) % 5 == 0
                or step + 1 == base_steps
            ):
                print(
                    f"Step {step+1}/{base_steps} | "
                    f"Loss={loss.item():.4f} | "
                    f"Base={base_loss.item():.4f} | "
                    f"TROHN={trohn_loss.item():.4f}",
                    flush=True
                )

        try:
            next(trohn_iter)
            raise RuntimeError("Unused TROHN data remains!")
        except StopIteration:
            pass

        # Full validation
        i2t, t2i = evaluate_retrieval(
            image_adapter, text_adapter
        )

        trohn_i2t, trohn_t2i, trohn_group = evaluate_trohn(
            image_adapter, text_adapter
        )

        avg_loss = total_loss / base_steps
        avg_base = total_base / base_steps
        avg_trohn = total_trohn / base_steps

        print(f"\nEpoch {epoch} completed")
        print(
            f"Loss={avg_loss:.4f} | "
            f"Base={avg_base:.4f} | "
            f"TROHN={avg_trohn:.4f}"
        )

        print(
            f"Flickr I2T@1={i2t:.2%} | "
            f"T2I@1={t2i:.2%}"
        )

        print(
            f"TROHN I2T={trohn_i2t:.2%} | "
            f"T2I={trohn_t2i:.2%} | "
            f"Group={trohn_group:.2%}"
        )

        result = {
            "image_adapter": image_adapter.state_dict(),
            "text_adapter": text_adapter.state_dict(),
            "alpha": ALPHA,
            "epoch": epoch,
            "use_trohn_img": True,
            "trohn_weight": TROHN_WEIGHT,
            "trohn_batch": TROHN_BATCH,
            "temperature": TEMPERATURE,
            "train_rows": train_rows,
            "validation_rows": val_rows,
            "flickr_i2t": i2t,
            "flickr_t2i": t2i,
            "trohn_i2t": trohn_i2t,
            "trohn_t2i": trohn_t2i,
            "trohn_group": trohn_group,
            "selection_metric": "Flickr8k I2T@1 + T2I@1",
        }

        torch.save(result, latest_path)

        with history_path.open(
            "a", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, avg_loss, avg_base, avg_trohn,
                i2t, t2i,
                trohn_i2t, trohn_t2i, trohn_group
            ])

        # Same checkpoint-selection rule as E2-A
        score = i2t + t2i

        if score > best_score:
            best_score = score
            torch.save(result, best_path)
            print("New best checkpoint saved!")

        print("Latest:", latest_path)

    print("\n========== E2-B TRAINING COMPLETE ==========")
    print("Best checkpoint:", best_path)
    print("Latest checkpoint:", latest_path)
    print("History:", history_path)


if __name__ == "__main__":
    train()
