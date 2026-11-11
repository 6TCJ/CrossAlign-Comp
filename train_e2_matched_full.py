#!/usr/bin/env python3
"""
CrossAlign-Comp: matched full-data ablation for E2-B.

Variants:
  1) full_control:      Flickr8k contrastive loss only
  2) full_text_only:    Flickr8k loss + TROHN hard-negative text objective
  3) full_text_image:   Flickr8k loss + original E2-B paired image/text objective

Every TROHN training row is visited once per epoch. All variants use the same
E1 initialization, TROHN row order, Flickr batches, optimizer, and validation.
Run without --train for a read-only preflight check.
"""

import argparse
import csv
import gc
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Paths/config -----------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "e2_data"
FULL_DATA = DATA / "e2_b_full"
TRAIN_SHARDS = FULL_DATA / "train"
VAL_SHARDS = FULL_DATA / "validation"

SEED = 42
EPOCHS = 5
FLICKR_BATCH_SIZE = 128
TROHN_BATCH_SIZE = 128
LR = 2e-4

ALPHA = 0.05
TEMPERATURE = 0.07
TROHN_WEIGHT = 0.5

EXPECTED_TRAIN_ROWS = 236_856
EXPECTED_VAL_ROWS = 59_214
EXPECTED_TRAIN_SHARDS = 58
EXPECTED_VAL_SHARDS = 15
FEATURE_DIM = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REQUIRED_SHARD_KEYS = (
    "image_features",
    "negative_image_features",
    "positive_features",
    "negative_features",
    "image_ids",
    "row_indices",
)

VARIANTS = (
    ("full_control", "control"),
    ("full_text_only", "text_only"),
    ("full_text_image", "text_image"),
)


# ----------------------------- Reproducibility -----------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- Model -----------------------------
class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def adapt(x, adapter):
    return F.normalize(x + ALPHA * adapter(x), dim=-1)


def load_pt(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def build_adapters(checkpoint):
    image_state = checkpoint["image_adapter"]
    text_state = checkpoint["text_adapter"]

    image_adapter = Adapter(
        FEATURE_DIM, image_state["fc1.weight"].shape[0]
    ).to(DEVICE)
    text_adapter = Adapter(
        FEATURE_DIM, text_state["fc1.weight"].shape[0]
    ).to(DEVICE)

    image_adapter.load_state_dict(image_state, strict=True)
    text_adapter.load_state_dict(text_state, strict=True)
    return image_adapter, text_adapter


# ----------------------------- Losses -----------------------------
def contrastive_loss(image_f, text_f):
    """Symmetric InfoNCE, with diagonal pairs as positives."""
    logits = image_f @ text_f.T / TEMPERATURE
    labels = torch.arange(len(image_f), device=DEVICE)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)


def trohn_text_only_loss(original_image_f, positive_text_f, negative_text_f):
    """
    Text-only hard-negative objective:
      - original image must retrieve its positive caption among positive and
        negative captions;
      - positive captions must retrieve their corresponding original images.
    No generated negative-image feature is used in this variant.
    """
    sim_pos = original_image_f @ positive_text_f.T / TEMPERATURE
    sim_neg = original_image_f @ negative_text_f.T / TEMPERATURE

    image_to_text_logits = torch.cat([sim_pos, sim_neg], dim=1)
    labels = torch.arange(len(original_image_f), device=DEVICE)

    loss_i2t = F.cross_entropy(image_to_text_logits, labels)
    loss_t2i = F.cross_entropy(sim_pos.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)


# ----------------------------- Shard loading/checks -----------------------------
def as_id_list(values):
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    return [str(x) for x in values]


def load_shard_split(split_name, folder, expected_rows, expected_shards):
    files = sorted(folder.glob("*.pt"))
    if not files:
        raise FileNotFoundError(
            f"No .pt shards found in {folder}\n"
            f"Expected full-data folder: {folder}"
        )

    if len(files) != expected_shards:
        raise RuntimeError(
            f"{split_name}: found {len(files)} shards, expected "
            f"{expected_shards}. Check the shard directory before training."
        )

    feature_parts = {key: [] for key in REQUIRED_SHARD_KEYS[:4]}
    all_image_ids = []
    total_rows = 0

    print(f"\nLoading {split_name} shards from: {folder}")
    for shard_no, path in enumerate(files, start=1):
        obj = load_pt(path)

        missing = [key for key in REQUIRED_SHARD_KEYS if key not in obj]
        if missing:
            raise KeyError(f"{path.name} is missing keys: {missing}")

        shard_n = len(obj["image_ids"])
        if len(obj["row_indices"]) != shard_n:
            raise RuntimeError(
                f"{path.name}: row_indices/image_ids length mismatch"
            )

        for key in REQUIRED_SHARD_KEYS[:4]:
            tensor = obj[key]
            if not torch.is_tensor(tensor):
                raise TypeError(f"{path.name}: {key} is not a tensor")
            if tensor.ndim != 2 or tensor.shape != (shard_n, FEATURE_DIM):
                raise RuntimeError(
                    f"{path.name}: {key} shape {tuple(tensor.shape)}, "
                    f"expected ({shard_n}, {FEATURE_DIM})"
                )
            feature_parts[key].append(tensor.float().cpu())

        all_image_ids.extend(as_id_list(obj["image_ids"]))
        total_rows += shard_n

        if shard_no == 1 or shard_no == len(files) or shard_no % 10 == 0:
            print(
                f"  {split_name}: {shard_no:02d}/{len(files)} shards, "
                f"{total_rows:,} rows loaded"
            )

        del obj

    if total_rows != expected_rows:
        raise RuntimeError(
            f"{split_name}: loaded {total_rows:,} rows; expected "
            f"{expected_rows:,}. Do not train on an incomplete split."
        )

    data = {
        key: torch.cat(parts, dim=0)
        for key, parts in feature_parts.items()
    }

    for key, tensor in data.items():
        if tensor.shape != (expected_rows, FEATURE_DIM):
            raise RuntimeError(
                f"{split_name}: final {key} shape is {tuple(tensor.shape)}"
            )

    print(
        f"{split_name} complete: {total_rows:,} rows / "
        f"{len(files)} shards"
    )
    return data, all_image_ids


def preflight():
    print("=" * 72)
    print("CrossAlign matched full-data ablation — PREFLIGHT")
    print("=" * 72)
    print("Project root:", ROOT)
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        print("WARNING: CUDA is unavailable; do not start full training on CPU.")

    for path in (
        ROOT / "clip_train_features.pt",
        ROOT / "clip_dev_features.pt",
        ROOT / "clip_adapter_e1.pt",
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        print("[OK]", path.relative_to(ROOT))

    train_data, train_ids = load_shard_split(
        "TRAIN", TRAIN_SHARDS, EXPECTED_TRAIN_ROWS, EXPECTED_TRAIN_SHARDS
    )
    val_data, val_ids = load_shard_split(
        "VALIDATION", VAL_SHARDS, EXPECTED_VAL_ROWS, EXPECTED_VAL_SHARDS
    )

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    overlap = train_id_set.intersection(val_id_set)
    if overlap:
        examples = sorted(overlap)[:5]
        raise RuntimeError(
            "Image-ID leakage detected between TROHN train and validation. "
            f"Example overlapping IDs: {examples}"
        )

    print("Train unique image IDs:", f"{len(train_id_set):,}")
    print("Validation unique image IDs:", f"{len(val_id_set):,}")
    print("Train/validation image-ID overlap: 0")

    train_cache = load_pt(ROOT / "clip_train_features.pt")
    dev_cache = load_pt(ROOT / "clip_dev_features.pt")
    e1_checkpoint = load_pt(ROOT / "clip_adapter_e1.pt")

    expected_base_shapes = {
        "train image_features": (6000, FEATURE_DIM),
        "train text_features": (30000, FEATURE_DIM),
        "dev image_features": (1000, FEATURE_DIM),
        "dev text_features": (5000, FEATURE_DIM),
    }
    actual_base_shapes = {
        "train image_features": tuple(train_cache["image_features"].shape),
        "train text_features": tuple(train_cache["text_features"].shape),
        "dev image_features": tuple(dev_cache["image_features"].shape),
        "dev text_features": tuple(dev_cache["text_features"].shape),
    }
    for key, expected in expected_base_shapes.items():
        actual = actual_base_shapes[key]
        if actual != expected:
            raise RuntimeError(f"{key}: {actual}, expected {expected}")
        print("[OK]", key, actual)

    if "image_adapter" not in e1_checkpoint or "text_adapter" not in e1_checkpoint:
        raise KeyError("E1 checkpoint does not contain image_adapter/text_adapter")

    print("\nExisting checkpoints:")
    for name in (
        "e2_a_control.pt",
        "e2_a_hard.pt",
        "e2_b_trohn_img_full_best.pt",
        "e2_b_trohn_img_full_latest.pt",
    ):
        path = DATA / name
        print((" [FOUND] " if path.exists() else " [not found] "), path)

    print("\nPREFLIGHT PASSED.")
    print("No model weights have been modified.")
    return train_data, val_data, train_cache, dev_cache, e1_checkpoint


# ----------------------------- Validation -----------------------------
@torch.no_grad()
def evaluate_retrieval(dev_images, dev_texts, image_adapter, text_adapter):
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
def evaluate_trohn(val_data, image_adapter, text_adapter):
    image_adapter.eval()
    text_adapter.eval()

    i0 = adapt(val_data["image_features"], image_adapter)
    i1 = adapt(val_data["negative_image_features"], image_adapter)
    t0 = adapt(val_data["positive_features"], text_adapter)
    t1 = adapt(val_data["negative_features"], text_adapter)

    s00 = (i0 * t0).sum(dim=-1)
    s01 = (i0 * t1).sum(dim=-1)
    s10 = (i1 * t0).sum(dim=-1)
    s11 = (i1 * t1).sum(dim=-1)

    i2t = ((s00 > s01) & (s11 > s10)).float().mean().item()
    t2i = ((s00 > s10) & (s11 > s01)).float().mean().item()
    group = (
        (s00 > s01) & (s11 > s10) &
        (s00 > s10) & (s11 > s01)
    ).float().mean().item()

    return i2t, t2i, group


# ----------------------------- Matched samplers -----------------------------
class FlickrBatchSampler:
    """Same deterministic Flickr batches for every ablation variant."""

    def __init__(self, n_images, seed):
        self.n_images = n_images
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(n_images, generator=self.generator)
        self.cursor = 0

    def next(self, batch_size):
        # Start a new permutation rather than allowing duplicate images
        # inside one contrastive batch.
        if self.cursor + batch_size > self.n_images:
            self.order = torch.randperm(
                self.n_images, generator=self.generator
            )
            self.cursor = 0

        image_ids = self.order[self.cursor:self.cursor + batch_size]
        self.cursor += batch_size

        caption_choices = torch.randint(
            0, 5, (batch_size,), generator=self.generator
        )
        caption_ids = image_ids * 5 + caption_choices
        return image_ids, caption_ids


# ----------------------------- Training -----------------------------
def move_feature_dict(data):
    return {
        key: F.normalize(value.float(), dim=-1).to(DEVICE)
        for key, value in data.items()
    }


def train_variant(
    name,
    mode,
    train_data,
    train_images,
    train_texts,
    dev_images,
    dev_texts,
    val_data,
    e1_checkpoint,
    history_writer,
    history_file,
):
    print("\n" + "=" * 72)
    print(f"TRAINING: {name}  |  mode={mode}")
    print("=" * 72)

    set_seed(SEED)
    image_adapter, text_adapter = build_adapters(e1_checkpoint)

    optimizer = torch.optim.AdamW(
        list(image_adapter.parameters()) +
        list(text_adapter.parameters()),
        lr=LR,
    )

    # Separate generators guarantee the same TROHN and Flickr samples
    # across all three variants, regardless of the loss branch.
    trohn_generator = torch.Generator().manual_seed(SEED)
    flickr_sampler = FlickrBatchSampler(
        len(train_images), seed=SEED + 10_000
    )

    n_trohn = len(train_data["image_features"])
    steps_per_epoch = math.ceil(n_trohn / TROHN_BATCH_SIZE)

    best_score = float("-inf")
    best_result = None

    best_path = DATA / f"e2_matched_{name}_best.pt"
    latest_path = DATA / f"e2_matched_{name}_latest.pt"

    for epoch in range(EPOCHS):
        image_adapter.train()
        text_adapter.train()

        trohn_order = torch.randperm(
            n_trohn, generator=trohn_generator
        )

        total_loss = 0.0
        total_base = 0.0
        total_trohn = 0.0

        for step, start in enumerate(
            range(0, n_trohn, TROHN_BATCH_SIZE), start=1
        ):
            trohn_ids_cpu = trohn_order[start:start + TROHN_BATCH_SIZE]
            trohn_ids = trohn_ids_cpu.to(DEVICE)

            flickr_ids_cpu, caption_ids_cpu = flickr_sampler.next(
                FLICKR_BATCH_SIZE
            )
            flickr_ids = flickr_ids_cpu.to(DEVICE)
            caption_ids = caption_ids_cpu.to(DEVICE)

            base_image_f = adapt(
                train_images[flickr_ids], image_adapter
            )
            base_text_f = adapt(
                train_texts[caption_ids], text_adapter
            )
            base_loss = contrastive_loss(base_image_f, base_text_f)

            if mode == "control":
                trohn_loss = torch.zeros((), device=DEVICE)

            else:
                i0 = adapt(
                    train_data["image_features"][trohn_ids],
                    image_adapter,
                )
                t0 = adapt(
                    train_data["positive_features"][trohn_ids],
                    text_adapter,
                )
                t1 = adapt(
                    train_data["negative_features"][trohn_ids],
                    text_adapter,
                )

                if mode == "text_only":
                    trohn_loss = trohn_text_only_loss(i0, t0, t1)

                elif mode == "text_image":
                    i1 = adapt(
                        train_data["negative_image_features"][trohn_ids],
                        image_adapter,
                    )
                    paired_images = torch.cat([i0, i1], dim=0)
                    paired_texts = torch.cat([t0, t1], dim=0)
                    trohn_loss = contrastive_loss(
                        paired_images, paired_texts
                    )

                else:
                    raise ValueError(f"Unknown training mode: {mode}")

            loss = base_loss + TROHN_WEIGHT * trohn_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_base += base_loss.item()
            total_trohn += trohn_loss.item()

            if step % 200 == 0 or step == steps_per_epoch:
                print(
                    f"{name} | Epoch {epoch + 1}/{EPOCHS} | "
                    f"Step {step}/{steps_per_epoch} | "
                    f"Loss={total_loss / step:.4f} | "
                    f"Base={total_base / step:.4f} | "
                    f"TROHN={total_trohn / step:.4f}"
                )

        flickr_i2t, flickr_t2i = evaluate_retrieval(
            dev_images, dev_texts, image_adapter, text_adapter
        )
        trohn_i2t, trohn_t2i, trohn_group = evaluate_trohn(
            val_data, image_adapter, text_adapter
        )

        mean_loss = total_loss / steps_per_epoch
        mean_base = total_base / steps_per_epoch
        mean_trohn = total_trohn / steps_per_epoch
        score = flickr_i2t + flickr_t2i

        result = {
            "variant": name,
            "epoch": epoch + 1,
            "train_loss": mean_loss,
            "base_loss": mean_base,
            "trohn_loss": mean_trohn,
            "flickr_i2t": flickr_i2t,
            "flickr_t2i": flickr_t2i,
            "trohn_i2t": trohn_i2t,
            "trohn_t2i": trohn_t2i,
            "trohn_group": trohn_group,
            "selection_score": score,
        }

        print(
            f"\n[{name}] Epoch {epoch + 1} summary | "
            f"Flickr={flickr_i2t:.2%}/{flickr_t2i:.2%} | "
            f"TROHN={trohn_i2t:.2%}/{trohn_t2i:.2%} | "
            f"Group={trohn_group:.2%}"
        )

        history_writer.writerow(result)
        history_file.flush()

        checkpoint_payload = {
            "image_adapter": image_adapter.state_dict(),
            "text_adapter": text_adapter.state_dict(),
            "alpha": ALPHA,
            "epoch": epoch + 1,
            "variant": name,
            "trohn_objective": mode,
            "trohn_weight": TROHN_WEIGHT,
            "trohn_batch_size": TROHN_BATCH_SIZE,
            "flickr_batch_size": FLICKR_BATCH_SIZE,
            "temperature": TEMPERATURE,
            "learning_rate": LR,
            "selection_metric": "Flickr8k I2T@1 + T2I@1",
            "train_rows": n_trohn,
            "validation_rows": len(val_data["image_features"]),
            "metrics": result,
        }

        # Always keep the latest epoch separately.
        torch.save(checkpoint_payload, latest_path)

        if score > best_score:
            best_score = score
            best_result = result.copy()
            torch.save(checkpoint_payload, best_path)
            print("[BEST UPDATED]", best_path)

    print(f"\nFinished {name}.")
    print("Best:", best_result)
    print("Best checkpoint:", best_path)
    print("Latest checkpoint:", latest_path)

    del image_adapter, text_adapter, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_path


# ----------------------------- Final comparison -----------------------------
@torch.no_grad()
def evaluate_checkpoint(path, dev_images, dev_texts, val_data):
    checkpoint = load_pt(path)
    image_adapter, text_adapter = build_adapters(checkpoint)

    flickr_i2t, flickr_t2i = evaluate_retrieval(
        dev_images, dev_texts, image_adapter, text_adapter
    )
    trohn_i2t, trohn_t2i, trohn_group = evaluate_trohn(
        val_data, image_adapter, text_adapter
    )

    return {
        "model": path.stem,
        "Flickr_I2T@1": flickr_i2t,
        "Flickr_T2I@1": flickr_t2i,
        "TROHN_I2T": trohn_i2t,
        "TROHN_T2I": trohn_t2i,
        "TROHN_Group": trohn_group,
    }


def compare_all(dev_images, dev_texts, val_data):
    candidates = [
        ("E1", ROOT / "clip_adapter_e1.pt"),
        ("E2-A_Control", DATA / "e2_a_control.pt"),
        ("E2-A_Hard", DATA / "e2_a_hard.pt"),
        ("E2-B_Full_Previous", DATA / "e2_b_trohn_img_full_best.pt"),
        ("Matched_Full_Control", DATA / "e2_matched_full_control_best.pt"),
        ("Matched_Full_TextOnly", DATA / "e2_matched_full_text_only_best.pt"),
        ("Matched_Full_TextImage", DATA / "e2_matched_full_text_image_best.pt"),
    ]

    results = []
    for label, path in candidates:
        if not path.exists():
            print("[SKIP missing checkpoint]", path)
            continue

        row = evaluate_checkpoint(
            path, dev_images, dev_texts, val_data
        )
        row["model"] = label
        results.append(row)

        print(
            f"{label:26s} | "
            f"Flickr {row['Flickr_I2T@1']:.2%}/{row['Flickr_T2I@1']:.2%} | "
            f"TROHN {row['TROHN_I2T']:.2%}/{row['TROHN_T2I']:.2%} | "
            f"Group {row['TROHN_Group']:.2%}"
        )

    if results:
        output = DATA / "e2_matched_full_comparison.csv"
        with output.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print("\nComparison saved:", output)


# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training after all preflight checks pass. Without this flag, only checks data.",
    )
    args = parser.parse_args()

    loaded = preflight()
    if not args.train:
        print("\nCHECK-ONLY MODE. To start training, run:")
        print("python train_e2_matched_full.py --train")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Full-data training is configured for the RTX 3090."
        )

    train_data_cpu, val_data_cpu, train_cache, dev_cache, e1_checkpoint = loaded

    print("\nMoving cached features to", DEVICE)
    train_data = move_feature_dict(train_data_cpu)
    val_data = move_feature_dict(val_data_cpu)

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

    del train_data_cpu, val_data_cpu, train_cache, dev_cache
    gc.collect()
    torch.cuda.empty_cache()

    history_path = DATA / "e2_matched_full_history.csv"
    history_fields = [
        "variant", "epoch", "train_loss", "base_loss", "trohn_loss",
        "flickr_i2t", "flickr_t2i", "trohn_i2t", "trohn_t2i",
        "trohn_group", "selection_score",
    ]

    with history_path.open("w", encoding="utf-8-sig", newline="") as history_file:
        history_writer = csv.DictWriter(
            history_file, fieldnames=history_fields
        )
        history_writer.writeheader()

        for name, mode in VARIANTS:
            train_variant(
                name=name,
                mode=mode,
                train_data=train_data,
                train_images=train_images,
                train_texts=train_texts,
                dev_images=dev_images,
                dev_texts=dev_texts,
                val_data=val_data,
                e1_checkpoint=e1_checkpoint,
                history_writer=history_writer,
                history_file=history_file,
            )

    print("\n=== Final matched comparison ===")
    compare_all(dev_images, dev_texts, val_data)

    print("\nAll matched full-data ablations completed.")
    print("History:", history_path)


if __name__ == "__main__":
    main()
