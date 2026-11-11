#!/usr/bin/env python3
"""CrossAlign-Comp E3-A: partial end-to-end CLIP fine-tuning.

This is a NEW training stage. It does not modify E1/E2 scripts, E2 feature
caches, or E2 checkpoints.

Initialization:
  - OpenCLIP ViT-B/32, pretrained=laion2b_s34b_b79k
  - Frozen E2 image/text Adapters from
    e2_data/continuation_full_text_image_v2/best_trohn_group.pt

Trainable CLIP parameters:
  - Last Transformer block of the visual encoder
  - Visual post-LayerNorm and visual projection
  - Last Transformer block of the text encoder
  - Text final LayerNorm and text projection
  - All other CLIP parameters are frozen
  - E2 Adapters are frozen in E3-A

Objective (same E2 full_text_image objective):
  total_loss = Flickr8k symmetric InfoNCE
             + 0.5 * paired TROHN image-text symmetric InfoNCE

Important:
  - This script reads raw Flickr8k/TROHN images and captions online.
  - It does NOT use cached 512-D CLIP features for training.
  - The clean TROHN validation split is reconstructed from the existing
    E2 shard image_ids/row_indices, retaining the 5,788 rows whose original
    image IDs do not occur in the full 236,856-row training split.
  - Default mode is check-only. Training requires --train.
  - latest.pt is atomically saved every 100 steps and at epoch completion.
  - --train --resume restores trainable CLIP weights, optimizer, AMP scaler,
    RNG states, Flickr sampler state, and the deterministic TROHN stream cursor.
  - At most save_every_steps - 1 optimizer steps can be repeated if interrupted
    after a periodic checkpoint.
  - Baseline E2 is included in model selection. Best TROHN Group and best Flickr
    checkpoints are tracked separately.

Run from /home/usst/jtc/crossalign-comp:
  python train_e3_partial_clip.py                         # check-only
  python train_e3_partial_clip.py --train --epochs 1      # new E3-A run
  python train_e3_partial_clip.py --train --resume --epochs 1

To extend the same run later:
  python train_e3_partial_clip.py --train --resume --epochs 2
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from datasets import DownloadConfig, load_dataset
from huggingface_hub import get_token
from PIL import Image


# ---------------------------------------------------------------------------
# Project paths and fixed experiment configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "e2_data"
COCO_DIR = DATA / "coco_full" / "train2017"
TRAIN_SHARDS = DATA / "e2_b_full" / "train"
VAL_SHARDS = DATA / "e2_b_full" / "validation"

SOURCE_E2_CHECKPOINT = (
    DATA / "continuation_full_text_image_v2" / "best_trohn_group.pt"
)
RUN_DIR = DATA / "e3_partial_clip_v1"

TROHN_DATASET = "imirandam/TROHN-Img"
FLICKR_DATASET = "intro/flickr8k"
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

SEED = 42
ALPHA_EXPECTED = 0.05
TEMPERATURE = 0.07
TROHN_WEIGHT = 0.5

# Conservative pilot before any full-data E3 run.
DEFAULT_EPOCHS = 1
DEFAULT_TRAIN_ROWS = 10_000
DEFAULT_LR = 1e-6
WEIGHT_DECAY = 0.01
TROHN_BATCH_SIZE = 16
FLICKR_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
FLICKR_IMAGE_EVAL_BATCH_SIZE = 32
FLICKR_TEXT_EVAL_BATCH_SIZE = 256
SAVE_EVERY_STEPS = 100
SHUFFLE_BUFFER_SIZE = 2048
GRAD_CLIP_NORM = 1.0

EXPECTED_TRAIN_ROWS = 236_856
EXPECTED_VAL_ROWS = 59_214
EXPECTED_TRAIN_SHARDS = 58
EXPECTED_VAL_SHARDS = 15
EXPECTED_CLEAN_VAL_ROWS = 5_788
CAPTIONS_PER_IMAGE = 5

RUN_VERSION = "e3_partial_clip_v1.0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Utilities and reproducibility
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_torch_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_pt(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def restore_rng_state(checkpoint):
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])


def get_hf_token():
    try:
        return get_token()
    except Exception:
        return None


def load_trohn_stream(split: str, shuffle_seed=None):
    kwargs = {"split": split, "streaming": True}
    token = get_hf_token()
    if token:
        kwargs["token"] = token
    dataset = load_dataset(TROHN_DATASET, **kwargs)
    if shuffle_seed is not None:
        dataset = dataset.shuffle(
            seed=int(shuffle_seed),
            buffer_size=SHUFFLE_BUFFER_SIZE,
        )
    return dataset


# ---------------------------------------------------------------------------
# Reconstruct the exact leakage-safe validation indices from E2 shards
# ---------------------------------------------------------------------------
def shard_signature(folder: Path, expected_shards: int):
    files = sorted(folder.glob("shard_*.pt"))
    if len(files) != expected_shards:
        raise RuntimeError(
            f"{folder}: found {len(files)} shard files; expected {expected_shards}"
        )
    return [
        {"name": p.name, "size": p.stat().st_size}
        for p in files
    ]


def load_clean_validation_indices():
    train_files = sorted(TRAIN_SHARDS.glob("shard_*.pt"))
    val_files = sorted(VAL_SHARDS.glob("shard_*.pt"))

    if len(train_files) != EXPECTED_TRAIN_SHARDS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_SHARDS} TRAIN shards, found {len(train_files)}"
        )
    if len(val_files) != EXPECTED_VAL_SHARDS:
        raise RuntimeError(
            f"Expected {EXPECTED_VAL_SHARDS} VALIDATION shards, found {len(val_files)}"
        )

    train_ids = set()
    train_rows = 0
    for path in train_files:
        obj = load_pt(path)
        if "image_ids" not in obj or "row_indices" not in obj:
            raise KeyError(f"{path.name} lacks image_ids/row_indices")
        ids = [str(x).strip() for x in obj["image_ids"]]
        row_indices = obj["row_indices"]
        if len(ids) != len(row_indices):
            raise RuntimeError(f"{path.name}: image_ids/row_indices length mismatch")
        train_ids.update(ids)
        train_rows += len(ids)
        del obj

    val_records = []
    val_rows = 0
    for path in val_files:
        obj = load_pt(path)
        if "image_ids" not in obj or "row_indices" not in obj:
            raise KeyError(f"{path.name} lacks image_ids/row_indices")
        ids = [str(x).strip() for x in obj["image_ids"]]
        row_indices = obj["row_indices"]
        if len(ids) != len(row_indices):
            raise RuntimeError(f"{path.name}: image_ids/row_indices length mismatch")
        if torch.is_tensor(row_indices):
            row_indices = row_indices.tolist()
        val_records.extend(
            (int(row_idx), image_id)
            for row_idx, image_id in zip(row_indices, ids)
        )
        val_rows += len(ids)
        del obj

    if train_rows != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            f"TRAIN shard rows={train_rows:,}; expected {EXPECTED_TRAIN_ROWS:,}"
        )
    if val_rows != EXPECTED_VAL_ROWS:
        raise RuntimeError(
            f"VALIDATION shard rows={val_rows:,}; expected {EXPECTED_VAL_ROWS:,}"
        )

    val_indices = [row_idx for row_idx, image_id in val_records if image_id not in train_ids]
    val_ids = [image_id for row_idx, image_id in val_records if image_id not in train_ids]

    if len(set(row_idx for row_idx, _ in val_records)) != EXPECTED_VAL_ROWS:
        raise RuntimeError("Validation row_indices are not unique")
    if min(row_idx for row_idx, _ in val_records) != 0:
        raise RuntimeError("Validation row_indices do not start at zero")
    if max(row_idx for row_idx, _ in val_records) != EXPECTED_VAL_ROWS - 1:
        raise RuntimeError("Validation row_indices do not cover the expected range")
    if len(val_indices) != EXPECTED_CLEAN_VAL_ROWS:
        raise RuntimeError(
            f"Clean validation rows={len(val_indices):,}; "
            f"expected {EXPECTED_CLEAN_VAL_ROWS:,}. Do not silently change the split."
        )
    if set(train_ids).intersection(val_ids):
        raise RuntimeError("Image-ID leakage remains in clean validation")

    val_indices.sort()
    return val_indices, {
        "train_rows": train_rows,
        "train_unique_image_ids": len(train_ids),
        "validation_rows": val_rows,
        "clean_validation_rows": len(val_indices),
        "clean_validation_unique_image_ids": len(set(val_ids)),
        "train_shards": shard_signature(TRAIN_SHARDS, EXPECTED_TRAIN_SHARDS),
        "validation_shards": shard_signature(VAL_SHARDS, EXPECTED_VAL_SHARDS),
    }


def data_signature(manifest):
    return sha256_json({
        "train_shards": manifest["train_shards"],
        "validation_shards": manifest["validation_shards"],
        "trohn_dataset": TROHN_DATASET,
        "flickr_dataset": FLICKR_DATASET,
    })


# ---------------------------------------------------------------------------
# Raw data decoding and batching
# ---------------------------------------------------------------------------
def decode_negative_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            from io import BytesIO
            with Image.open(BytesIO(value["bytes"])) as im:
                return im.convert("RGB").copy()
        if value.get("path"):
            with Image.open(value["path"]) as im:
                return im.convert("RGB").copy()
    raise TypeError(f"Cannot decode TROHN negative_image: {type(value)}")


def original_image_tensor(image_id, preprocess):
    image_id = str(image_id).strip()
    path = COCO_DIR / image_id
    if not path.exists():
        raise FileNotFoundError(f"Missing original COCO image: {path}")
    with Image.open(path) as im:
        return preprocess(im.convert("RGB"))


def trohn_batch_tensors(rows, preprocess, tokenizer):
    original_images = []
    negative_images = []
    positive_captions = []
    negative_captions = []

    for row in rows:
        image_id = str(row["image_id"]).strip()
        original_images.append(original_image_tensor(image_id, preprocess))
        negative_images.append(preprocess(decode_negative_image(row["negative_image"])))
        positive_captions.append(str(row["caption"]).strip())
        negative_captions.append(str(row["negative_caption"]).strip())

    if any(not x for x in positive_captions + negative_captions):
        raise RuntimeError("TROHN batch contains an empty caption")

    image_tensor = torch.cat([
        torch.stack(original_images),
        torch.stack(negative_images),
    ], dim=0)
    text_tensor = tokenizer(positive_captions + negative_captions)
    return image_tensor.to(DEVICE, non_blocking=True), text_tensor.to(DEVICE, non_blocking=True)


def flickr_batch_tensors(dataset, image_ids, caption_ids, preprocess, tokenizer):
    images = []
    captions = []
    for image_idx, caption_idx in zip(image_ids.tolist(), caption_ids.tolist()):
        row = dataset[int(image_idx)]
        images.append(preprocess(row["image"].convert("RGB")))
        captions.append(str(row[f"caption_{int(caption_idx)}"]).strip())
    if any(not x for x in captions):
        raise RuntimeError("Flickr batch contains an empty caption")
    return (
        torch.stack(images).to(DEVICE, non_blocking=True),
        tokenizer(captions).to(DEVICE, non_blocking=True),
    )


class FlickrBatchSampler:
    """Deterministic no-duplicate-image batches, with resumable sampler state."""

    def __init__(self, n_images, seed):
        self.n_images = int(n_images)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.order = torch.randperm(self.n_images, generator=self.generator)
        self.cursor = 0

    def next(self, batch_size):
        if self.cursor + batch_size > self.n_images:
            self.order = torch.randperm(self.n_images, generator=self.generator)
            self.cursor = 0
        image_ids = self.order[self.cursor:self.cursor + batch_size]
        self.cursor += batch_size
        caption_ids = torch.randint(
            0, CAPTIONS_PER_IMAGE, (batch_size,), generator=self.generator
        )
        return image_ids, caption_ids

    def state_dict(self):
        return {
            "n_images": self.n_images,
            "generator_state": self.generator.get_state(),
            "order": self.order.clone(),
            "cursor": int(self.cursor),
        }

    def load_state_dict(self, state):
        if int(state["n_images"]) != self.n_images:
            raise RuntimeError("Flickr sampler image count changed")
        order = state["order"].long().cpu()
        if order.numel() != self.n_images:
            raise RuntimeError("Flickr sampler order has an invalid length")
        cursor = int(state["cursor"])
        if not 0 <= cursor <= self.n_images:
            raise RuntimeError("Flickr sampler cursor is invalid")
        self.generator.set_state(state["generator_state"])
        self.order = order
        self.cursor = cursor
    def collect_rows(iterator, count):
        rows = []
        for _ in range(count):
            try:
                rows.append(next(iterator))
            except StopIteration:
                break
        return rows


# ---------------------------------------------------------------------------
# CLIP and frozen E2 Adapter construction
# ---------------------------------------------------------------------------
class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def unfreeze_module_or_parameter(obj):
    if isinstance(obj, nn.Module):
        for parameter in obj.parameters():
            parameter.requires_grad_(True)
    elif isinstance(obj, nn.Parameter):
        obj.requires_grad_(True)
    elif obj is not None:
        raise TypeError(f"Unsupported trainable CLIP component: {type(obj)}")


def build_clip_and_unfreeze_last_blocks():
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
    )
    model = model.to(DEVICE)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    visual_blocks = getattr(getattr(model.visual, "transformer", None), "resblocks", None)
    text_blocks = getattr(getattr(model, "transformer", None), "resblocks", None)
    if visual_blocks is None or text_blocks is None:
        raise RuntimeError(
            "Unexpected OpenCLIP ViT-B/32 architecture: transformer.resblocks not found"
        )
    if len(visual_blocks) < 1 or len(text_blocks) < 1:
        raise RuntimeError("CLIP transformer has no residual blocks")

    # Last transformer block in each modality.
    unfreeze_module_or_parameter(visual_blocks[-1])
    unfreeze_module_or_parameter(text_blocks[-1])

    # Output normalization and projection layers are trainable as well.
    unfreeze_module_or_parameter(getattr(model.visual, "ln_post", None))
    unfreeze_module_or_parameter(getattr(model.visual, "proj", None))
    unfreeze_module_or_parameter(getattr(model, "ln_final", None))
    unfreeze_module_or_parameter(getattr(model, "text_projection", None))

    # Keep CLIP in eval mode (no stochastic dropout); requires_grad still works
    # for the explicitly unfrozen parameters.
    model.eval()

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No CLIP parameters were unfrozen")

    required_fragments = (
        f"visual.transformer.resblocks.{len(visual_blocks) - 1}",
        f"transformer.resblocks.{len(text_blocks) - 1}",
        "visual.ln_post",
        "visual.proj",
        "ln_final",
        "text_projection",
    )
    trainable_names = [name for name, _ in trainable]
    for fragment in required_fragments:
        if not any(name.startswith(fragment) for name in trainable_names):
            raise RuntimeError(
                f"Expected trainable component '{fragment}' not found. "
                f"Actual trainable names: {trainable_names[:20]}"
            )

    return model, preprocess, trainable


def build_frozen_adapters(e2_checkpoint):
    if "image_adapter" not in e2_checkpoint or "text_adapter" not in e2_checkpoint:
        raise KeyError("E2 checkpoint lacks image_adapter/text_adapter")

    alpha = float(e2_checkpoint.get("alpha", -1))
    if abs(alpha - ALPHA_EXPECTED) > 1e-12:
        raise RuntimeError(
            f"E2 Adapter alpha={alpha}; expected {ALPHA_EXPECTED}. "
            "Do not silently mix Adapter scales."
        )

    image_state = e2_checkpoint["image_adapter"]
    text_state = e2_checkpoint["text_adapter"]
    image_hidden = int(image_state["fc1.weight"].shape[0])
    text_hidden = int(text_state["fc1.weight"].shape[0])
    image_adapter = Adapter(512, image_hidden).to(DEVICE)
    text_adapter = Adapter(512, text_hidden).to(DEVICE)
    image_adapter.load_state_dict(image_state, strict=True)
    text_adapter.load_state_dict(text_state, strict=True)

    for parameter in image_adapter.parameters():
        parameter.requires_grad_(False)
    for parameter in text_adapter.parameters():
        parameter.requires_grad_(False)
    image_adapter.eval()
    text_adapter.eval()
    return image_adapter, text_adapter, alpha


def adapt(features, adapter, alpha):
    return F.normalize(features + alpha * adapter(features), dim=-1)



# ---------------------------------------------------------------------------
# Losses and metrics (kept consistent with E2)
# ---------------------------------------------------------------------------
def contrastive_loss(image_features, text_features):
    logits = image_features @ text_features.T / TEMPERATURE
    labels = torch.arange(len(image_features), device=DEVICE)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)


def trohn_metrics_from_features(i0, i1, t0, t1):
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


@torch.no_grad()
def evaluate_flickr(model, image_adapter, text_adapter, dataset, preprocess, tokenizer, alpha):
    model.eval()
    image_parts = []
    captions = []

    for start in range(0, len(dataset), FLICKR_IMAGE_EVAL_BATCH_SIZE):
        rows = [
            dataset[i]
            for i in range(start, min(start + FLICKR_IMAGE_EVAL_BATCH_SIZE, len(dataset)))
        ]
        images = torch.stack([
            preprocess(row["image"].convert("RGB")) for row in rows
        ]).to(DEVICE, non_blocking=True)
        for row in rows:
            for caption_idx in range(CAPTIONS_PER_IMAGE):
                captions.append(str(row[f"caption_{caption_idx}"]).strip())

        # Evaluation stays in FP32 to match the original E2 feature-extraction
        # and validation protocol as closely as possible.
        features = F.normalize(model.encode_image(images).float(), dim=-1)
        image_parts.append(adapt(features, image_adapter, alpha))

    text_parts = []
    for start in range(0, len(captions), FLICKR_TEXT_EVAL_BATCH_SIZE):
        tokens = tokenizer(captions[start:start + FLICKR_TEXT_EVAL_BATCH_SIZE]).to(DEVICE)
        features = F.normalize(model.encode_text(tokens).float(), dim=-1)
        text_parts.append(adapt(features, text_adapter, alpha))

    image_features = torch.cat(image_parts, dim=0)
    text_features = torch.cat(text_parts, dim=0)
    similarity = image_features @ text_features.T

    image_ids = torch.arange(len(image_features), device=DEVICE)
    text_image_ids = torch.arange(len(text_features), device=DEVICE) // CAPTIONS_PER_IMAGE
    i2t = (similarity.argmax(dim=1) // CAPTIONS_PER_IMAGE == image_ids).float().mean().item()
    t2i = (similarity.argmax(dim=0) == text_image_ids).float().mean().item()
    return i2t, t2i


@torch.no_grad()
def evaluate_clean_trohn(model, image_adapter, text_adapter, clean_val_indices,
                         preprocess, tokenizer, alpha):
    model.eval()
    selected = set(clean_val_indices)
    iterator = iter(load_trohn_stream("validation"))

    i0_parts, i1_parts, t0_parts, t1_parts = [], [], [], []
    batch_rows = []
    found = 0
    last_index = max(clean_val_indices)

    for row_index, row in enumerate(iterator):
        if row_index in selected:
            batch_rows.append(row)
            found += 1

        if len(batch_rows) == VAL_BATCH_SIZE:
            images, tokens = trohn_batch_tensors(batch_rows, preprocess, tokenizer)
            image_f = F.normalize(model.encode_image(images).float(), dim=-1)
            text_f = F.normalize(model.encode_text(tokens).float(), dim=-1)
            n = len(batch_rows)
            image_f = adapt(image_f, image_adapter, alpha)
            text_f = adapt(text_f, text_adapter, alpha)
            i0_parts.append(image_f[:n].cpu())
            i1_parts.append(image_f[n:].cpu())
            t0_parts.append(text_f[:n].cpu())
            t1_parts.append(text_f[n:].cpu())
            batch_rows = []

        if row_index >= last_index:
            break

    if batch_rows:
        images, tokens = trohn_batch_tensors(batch_rows, preprocess, tokenizer)
        image_f = F.normalize(model.encode_image(images).float(), dim=-1)
        text_f = F.normalize(model.encode_text(tokens).float(), dim=-1)
        n = len(batch_rows)
        image_f = adapt(image_f, image_adapter, alpha)
        text_f = adapt(text_f, text_adapter, alpha)
        i0_parts.append(image_f[:n].cpu())
        i1_parts.append(image_f[n:].cpu())
        t0_parts.append(text_f[:n].cpu())
        t1_parts.append(text_f[n:].cpu())

    if found != len(clean_val_indices):
        raise RuntimeError(
            f"Read {found} clean validation rows; expected {len(clean_val_indices)}"
        )

    i0 = torch.cat(i0_parts).to(DEVICE)
    i1 = torch.cat(i1_parts).to(DEVICE)
    t0 = torch.cat(t0_parts).to(DEVICE)
    t1 = torch.cat(t1_parts).to(DEVICE)
    return trohn_metrics_from_features(i0, i1, t0, t1)


def evaluate_all(model, image_adapter, text_adapter, flickr_dev, clean_val_indices,
                 preprocess, tokenizer, alpha, label):
    flickr_i2t, flickr_t2i = evaluate_flickr(
        model, image_adapter, text_adapter, flickr_dev, preprocess, tokenizer, alpha
    )
    trohn_i2t, trohn_t2i, trohn_group = evaluate_clean_trohn(
        model, image_adapter, text_adapter, clean_val_indices,
        preprocess, tokenizer, alpha,
    )
    metrics = {
        "flickr_i2t": float(flickr_i2t),
        "flickr_t2i": float(flickr_t2i),
        "trohn_i2t": float(trohn_i2t),
        "trohn_t2i": float(trohn_t2i),
        "trohn_group": float(trohn_group),
    }
    print(
        f"{label:28s} | Flickr {flickr_i2t:.2%}/{flickr_t2i:.2%} | "
        f"TROHN {trohn_i2t:.2%}/{trohn_t2i:.2%} | Group {trohn_group:.2%}",
        flush=True,
    )
    return metrics


def trohn_selection_key(metrics):
    return (
        metrics["trohn_group"],
        (metrics["trohn_i2t"] + metrics["trohn_t2i"]) / 2.0,
    )


def flickr_selection_key(metrics):
    return (
        metrics["flickr_i2t"] + metrics["flickr_t2i"],
        metrics["trohn_group"],
    )


# ---------------------------------------------------------------------------
# Checkpoint state and model selection
# ---------------------------------------------------------------------------
def trainable_state_dict(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(model, state):
    parameters = dict(model.named_parameters())
    expected = {name for name, p in parameters.items() if p.requires_grad}
    received = set(state)
    if expected != received:
        raise RuntimeError(
            "Trainable CLIP parameter keys mismatch. "
            f"Missing={sorted(expected - received)}; extra={sorted(received - expected)}"
        )
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(device=parameters[name].device,
                                            dtype=parameters[name].dtype))


def adapter_state_dict(image_adapter, text_adapter):
    return {
        "image_adapter": {k: v.detach().cpu().clone() for k, v in image_adapter.state_dict().items()},
        "text_adapter": {k: v.detach().cpu().clone() for k, v in text_adapter.state_dict().items()},
    }


def inference_checkpoint_payload(model, image_adapter, text_adapter, alpha,
                                 metrics, epoch, label, source_hash, script_hash):
    return {
        "run_version": RUN_VERSION,
        "checkpoint_label": label,
        "epoch": int(epoch),
        "clip_model_name": CLIP_MODEL_NAME,
        "clip_pretrained": CLIP_PRETRAINED,
        "trainable_model_state": trainable_state_dict(model),
        **adapter_state_dict(image_adapter, text_adapter),
        "alpha": float(alpha),
        "metrics_clean_validation": metrics,
        "source_e2_checkpoint": str(SOURCE_E2_CHECKPOINT),
        "source_e2_checkpoint_sha256": source_hash,
        "runner_sha256": script_hash,
        "trainable_scope": "last visual block + visual ln_post/proj + last text block + ln_final/text_projection",
    }


def verify_baseline_matches_e2(baseline_metrics, e2_checkpoint):
    """Catch raw-data/preprocessing mismatch before any E3 optimizer step."""
    stored = e2_checkpoint.get("metrics_clean_validation")
    if not isinstance(stored, dict):
        print("[WARN] E2 checkpoint has no metrics_clean_validation; baseline cross-check skipped.", flush=True)
        return

    keys = ("flickr_i2t", "flickr_t2i", "trohn_i2t", "trohn_t2i", "trohn_group")
    differences = {}
    for key in keys:
        if key not in stored:
            print(f"[WARN] E2 checkpoint metric '{key}' missing; baseline cross-check skipped.", flush=True)
            return
        differences[key] = abs(float(baseline_metrics[key]) - float(stored[key]))

    print("E2 baseline re-evaluation absolute differences:", differences, flush=True)
    max_difference = max(differences.values())
    if max_difference > 0.01:
        raise RuntimeError(
            "Raw-data E2 baseline differs from the saved E2 clean-validation metrics "
            f"by {max_difference:.4f} (> 0.01). Stop before E3 training and inspect "
            "the TROHN row order, image preprocessing, Flickr split, and source checkpoint."
        )
    print("[OK] Raw-data E2 baseline matches saved E2 metrics within 1 percentage point.", flush=True)


def initialize_best_checkpoints(model, image_adapter, text_adapter, alpha,
                                baseline_metrics, source_epoch, source_hash,
                                script_hash, summary_path, config):
    summary = {
        "project": "CrossAlign-Comp",
        "run": RUN_VERSION,
        "stage": "E3-A partial CLIP fine-tuning; E2 Adapters frozen",
        "source_e2_checkpoint": str(SOURCE_E2_CHECKPOINT),
        "source_e2_epoch": int(source_epoch),
        "source_e2_checkpoint_sha256": source_hash,
        "clip_model": CLIP_MODEL_NAME,
        "clip_pretrained": CLIP_PRETRAINED,
        "alpha": float(alpha),
        "learning_rate": config["learning_rate"],
        "train_rows_per_epoch": config["train_rows_per_epoch"],
        "trohn_batch_size": TROHN_BATCH_SIZE,
        "flickr_batch_size": FLICKR_BATCH_SIZE,
        "trohn_weight": TROHN_WEIGHT,
        "temperature": TEMPERATURE,
        "clean_validation_rows": EXPECTED_CLEAN_VAL_ROWS,
        "selection_policy": {
            "primary": "max clean-validation TROHN_Group",
            "tie_break": "max mean(TROHN_I2T, TROHN_T2I)",
            "separate_guardrail": "max Flickr_I2T@1 + Flickr_T2I@1; tie-break TROHN_Group",
            "baseline_included": True,
        },
        "baseline_e2": baseline_metrics,
        "best_trohn_group": {
            "checkpoint": "best_trohn_group.pt",
            "epoch": int(source_epoch),
            "metrics": baseline_metrics,
            "selection_key": list(trohn_selection_key(baseline_metrics)),
        },
        "best_flickr": {
            "checkpoint": "best_flickr.pt",
            "epoch": int(source_epoch),
            "metrics": baseline_metrics,
            "selection_key": list(flickr_selection_key(baseline_metrics)),
                    },
    }

    baseline_trohn = inference_checkpoint_payload(
        model, image_adapter, text_adapter, alpha, baseline_metrics,
        source_epoch, "e2_baseline_best_trohn", source_hash, script_hash,
    )
    baseline_flickr = dict(baseline_trohn)
    baseline_flickr["checkpoint_label"] = "e2_baseline_best_flickr"
    atomic_torch_save(baseline_trohn, RUN_DIR / "best_trohn_group.pt")
    atomic_torch_save(baseline_flickr, RUN_DIR / "best_flickr.pt")
    atomic_json_save(summary, summary_path)
    return summary


def update_best_checkpoints(model, image_adapter, text_adapter, alpha, metrics,
                            global_epoch, summary, summary_path, source_hash,
                            script_hash):
    if trohn_selection_key(metrics) > tuple(summary["best_trohn_group"]["selection_key"]):
        payload = inference_checkpoint_payload(
            model, image_adapter, text_adapter, alpha, metrics, global_epoch,
            "best_trohn_group", source_hash, script_hash,
        )
        atomic_torch_save(payload, RUN_DIR / "best_trohn_group.pt")
        summary["best_trohn_group"] = {
            "checkpoint": "best_trohn_group.pt",
            "epoch": int(global_epoch),
            "metrics": metrics,
            "selection_key": list(trohn_selection_key(metrics)),
        }
        print("[BEST TROHN GROUP UPDATED]", RUN_DIR / "best_trohn_group.pt", flush=True)

    if flickr_selection_key(metrics) > tuple(summary["best_flickr"]["selection_key"]):
        payload = inference_checkpoint_payload(
            model, image_adapter, text_adapter, alpha, metrics, global_epoch,
            "best_flickr", source_hash, script_hash,
        )
        atomic_torch_save(payload, RUN_DIR / "best_flickr.pt")
        summary["best_flickr"] = {
            "checkpoint": "best_flickr.pt",
            "epoch": int(global_epoch),
            "metrics": metrics,
            "selection_key": list(flickr_selection_key(metrics)),
        }
        print("[BEST FLICKR UPDATED]", RUN_DIR / "best_flickr.pt", flush=True)

    atomic_json_save(summary, summary_path)


def write_history_row(path: Path, row):
    fields = [
        "epoch", "global_step", "train_loss", "flickr_loss", "trohn_loss",
        "flickr_i2t", "flickr_t2i", "trohn_i2t", "trohn_t2i", "trohn_group",
        "learning_rate", "train_rows",
    ]
    existing = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            existing = list(csv.DictReader(f))
    # Idempotent epoch write: if interruption occurs after history write but
    # before epoch-complete checkpoint, re-evaluation replaces that epoch row.
    existing = [r for r in existing if int(r["epoch"]) != int(row["epoch"])]
    existing.append({k: row.get(k, "") for k in fields})
    existing.sort(key=lambda r: int(r["epoch"]))

    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_latest_payload(model, optimizer, scaler, sampler, epoch,
                         step_in_epoch, rows_seen, completed_epochs,
                         global_step, total_loss, total_flickr, total_trohn,
                         status, config, source_hash, script_hash, manifest_hash):
    return {
        "run_version": RUN_VERSION,
        "status": status,
        "epoch_in_progress": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "rows_seen_in_epoch": int(rows_seen),
        "completed_epochs": int(completed_epochs),
        "global_step": int(global_step),
        "trainable_model_state": trainable_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "flickr_sampler_state": sampler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "total_loss": float(total_loss),
        "total_flickr": float(total_flickr),
        "total_trohn": float(total_trohn),
        "config": config,
        "source_e2_checkpoint_sha256": source_hash,
        "runner_sha256": script_hash,
        "data_manifest_sha256": manifest_hash,
    }


def validate_resume_checkpoint(checkpoint, config, source_hash, script_hash, manifest_hash):
    if checkpoint.get("run_version") != RUN_VERSION:
        raise RuntimeError("E3 run version mismatch; cannot exact-resume")
    if checkpoint.get("runner_sha256") != script_hash:
        raise RuntimeError("E3 script changed since checkpoint; restore matching script before resume")
    if checkpoint.get("source_e2_checkpoint_sha256") != source_hash:
        raise RuntimeError("E2 source checkpoint changed since E3 run started")
    if checkpoint.get("data_manifest_sha256") != manifest_hash:
        raise RuntimeError("E2 shard manifest changed since E3 run started")
    if checkpoint.get("config") != config:
        raise RuntimeError(
            "E3 training configuration changed. Exact resume requires the original "
            f"configuration. Saved={checkpoint.get('config')} Current={config}"
        )


# ---------------------------------------------------------------------------
# Preflight and gradient smoke test
# ---------------------------------------------------------------------------
def load_flickr_splits():
    download_config = DownloadConfig(max_retries=10)
    train = load_dataset(
        FLICKR_DATASET,
        data_files={"train": "train/**"},
        split="train",
        download_config=download_config,
    )
    dev = load_dataset(
        FLICKR_DATASET,
        data_files={"dev": "dev/**"},
        split="dev",
        download_config=download_config,
    )
    if len(train) != 6000 or len(dev) != 1000:
        raise RuntimeError(
            f"Unexpected Flickr split sizes: train={len(train)}, dev={len(dev)}"
        )
    required = [f"caption_{i}" for i in range(CAPTIONS_PER_IMAGE)]
    for column in ["image"] + required:
        if column not in train.column_names or column not in dev.column_names:
            raise RuntimeError(f"Flickr dataset missing column: {column}")
    return train, dev


def collect_rows(iterator, count):
    rows = []
    for _ in range(count):
        try:
            rows.append(next(iterator))
        except StopIteration:
            break
    return rows


def run_gradient_smoke_test(model, trainable, image_adapter, text_adapter,
                            alpha, preprocess, tokenizer, flickr_train):
    # A real forward/backward pass verifies that the selected CLIP blocks receive
    # gradients and that the RTX 3090 can hold the initial batch. No optimizer
    # step is executed, so model weights remain unchanged.
    trohn_dataset = load_trohn_stream("train")
    trohn_iterator = iter(trohn_dataset)

    try:
        trohn_rows = collect_rows(
            trohn_iterator,
            TROHN_BATCH_SIZE
        )
    finally:
        close_fn = getattr(trohn_iterator, "close", None)
        if callable(close_fn):
            close_fn()

        del trohn_iterator
        del trohn_dataset

        import gc
        gc.collect()
    if len(trohn_rows) != TROHN_BATCH_SIZE:
        raise RuntimeError("Could not read a complete TROHN smoke-test batch")

    sampler = FlickrBatchSampler(len(flickr_train), SEED + 10_000)
    flickr_ids, caption_ids = sampler.next(FLICKR_BATCH_SIZE)
    flickr_images, flickr_tokens = flickr_batch_tensors(
        flickr_train, flickr_ids, caption_ids, preprocess, tokenizer
    )
    trohn_images, trohn_tokens = trohn_batch_tensors(
        trohn_rows, preprocess, tokenizer
    )

    images = torch.cat([flickr_images, trohn_images], dim=0)
    tokens = torch.cat([flickr_tokens, trohn_tokens], dim=0)
    model.zero_grad(set_to_none=True)

    # Use FP32 for this one-batch gradient check. The actual training loop uses
    # AMP + GradScaler; FP32 here avoids mistaking an unscaled smoke-test
    # gradient overflow for a model-graph problem.
    image_features = F.normalize(model.encode_image(images).float(), dim=-1)
    text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)
    image_features = adapt(image_features, image_adapter, alpha)
    text_features = adapt(text_features, text_adapter, alpha)

    fb = FLICKR_BATCH_SIZE
    tb = TROHN_BATCH_SIZE
    flickr_loss = contrastive_loss(image_features[:fb], text_features[:fb])
    trohn_images_f = torch.cat([image_features[fb:fb + tb], image_features[fb + tb:]], dim=0)
    trohn_texts_f = torch.cat([text_features[fb:fb + tb], text_features[fb + tb:]], dim=0)
    trohn_loss = contrastive_loss(trohn_images_f, trohn_texts_f)
    loss = flickr_loss + TROHN_WEIGHT * trohn_loss

    loss.backward()
    grad_params = [p for _, p in trainable if p.grad is not None]
    if len(grad_params) != len(trainable):
        missing = [name for name, p in trainable if p.grad is None]
        raise RuntimeError(
            "Smoke test failed: some selected CLIP parameters received no gradient: "
            + ", ".join(missing[:20])
        )
    if not all(torch.isfinite(p.grad).all().item() for p in grad_params):
        raise RuntimeError("Smoke test failed: non-finite CLIP gradients")

    grad_norm = torch.sqrt(sum(p.grad.detach().float().pow(2).sum() for p in grad_params)).item()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    print(
        f"[OK] Gradient smoke test | loss={loss.item():.5f} | "
        f"trainable tensors with grad={len(grad_params)}/{len(trainable)} | "
        f"grad_norm={grad_norm:.5f}",
        flush=True,
    )


def preflight():
    print("=" * 76)
    print("CrossAlign-Comp E3-A Partial CLIP Fine-tuning — PREFLIGHT")
    print("=" * 76)
    print("Project root:", ROOT)
    print("Device:", DEVICE)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Expected server NVIDIA RTX 3090.")
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
    )

    required_paths = [
        COCO_DIR,
        TRAIN_SHARDS,
        VAL_SHARDS,
        SOURCE_E2_CHECKPOINT,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required E3 input missing: {path}")
        print("[OK]", path.relative_to(ROOT))

    clean_indices, manifest = load_clean_validation_indices()
    print("TRAIN rows:", f"{manifest['train_rows']:,}")
    print("TRAIN unique original image IDs:", f"{manifest['train_unique_image_ids']:,}")
    print("Original validation rows:", f"{manifest['validation_rows']:,}")
    print("Clean held-out validation rows:", f"{manifest['clean_validation_rows']:,}")
    print("Clean validation unique images:", f"{manifest['clean_validation_unique_image_ids']:,}")
    print("CLEAN train/validation image-ID overlap: 0")

    flickr_train, flickr_dev = load_flickr_splits()
    print("[OK] Flickr8k train/dev:", len(flickr_train), "/", len(flickr_dev))

    e2_checkpoint = load_pt(SOURCE_E2_CHECKPOINT)
    source_epoch = int(e2_checkpoint.get("epoch", -1))
    image_adapter, text_adapter, alpha = build_frozen_adapters(e2_checkpoint)
    print("[OK] E2 source checkpoint:", SOURCE_E2_CHECKPOINT)
    print("E2 source epoch:", source_epoch)
    print("E2 alpha:", alpha)
    if "metrics_clean_validation" in e2_checkpoint:
        print("E2 stored clean-validation metrics:", e2_checkpoint["metrics_clean_validation"])

    model, preprocess, trainable = build_clip_and_unfreeze_last_blocks()
    trainable_count = sum(p.numel() for _, p in trainable)
    total_count = sum(p.numel() for p in model.parameters())
    print("CLIP model:", f"{CLIP_MODEL_NAME} / {CLIP_PRETRAINED}")
    print("Trainable parameter tensors:", len(trainable))
    print("Trainable parameters:", f"{trainable_count:,}")
    print("Total CLIP parameters:", f"{total_count:,}")
    print("Trainable scope:")
    for name, parameter in trainable:
        print(f"  {name}: {tuple(parameter.shape)}")

    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    run_gradient_smoke_test(
        model, trainable, image_adapter, text_adapter, alpha,
        preprocess, tokenizer, flickr_train,
    )

    print("PREFLIGHT PASSED. No optimizer step or model-weight update was performed.")
    return {
        "clean_indices": clean_indices,
        "manifest": manifest,
        "flickr_train": flickr_train,
        "flickr_dev": flickr_dev,
        "e2_checkpoint": e2_checkpoint,
        "image_adapter": image_adapter,
        "text_adapter": text_adapter,
        "alpha": alpha,
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
        "trainable": trainable,
        "source_epoch": source_epoch,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_config(args, manifest_hash):
    return {
        "clip_model_name": CLIP_MODEL_NAME,
        "clip_pretrained": CLIP_PRETRAINED,
        "unfreeze_last_visual_blocks": 1,
        "unfreeze_last_text_blocks": 1,
        "unfreeze_visual_ln_post_projection": True,
        "unfreeze_text_ln_final_projection": True,
        "freeze_e2_adapters": True,
        "alpha": ALPHA_EXPECTED,
        "temperature": TEMPERATURE,
                "trohn_weight": TROHN_WEIGHT,
        "train_rows_per_epoch": int(args.train_rows),
        "trohn_batch_size": TROHN_BATCH_SIZE,
        "flickr_batch_size": FLICKR_BATCH_SIZE,
        "learning_rate": float(args.lr),
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "seed": SEED,
        "shuffle_buffer_size": SHUFFLE_BUFFER_SIZE,
        "data_manifest_sha256": manifest_hash,
    }


def save_latest(latest_path, model, optimizer, scaler, sampler,
                epoch, step_in_epoch, rows_seen, completed_epochs,
                global_step, total_loss, total_flickr, total_trohn,
                status, config, source_hash, script_hash, manifest_hash):
    payload = build_latest_payload(
        model, optimizer, scaler, sampler,
        epoch, step_in_epoch, rows_seen, completed_epochs,
        global_step, total_loss, total_flickr, total_trohn,
        status, config, source_hash, script_hash, manifest_hash,
    )
    atomic_torch_save(payload, latest_path)


def train_one_step(model, trainable, image_adapter, text_adapter, alpha,
                   optimizer, scaler, flickr_train, sampler, trohn_rows,
                   preprocess, tokenizer):
    flickr_ids, caption_ids = sampler.next(FLICKR_BATCH_SIZE)
    flickr_images, flickr_tokens = flickr_batch_tensors(
        flickr_train, flickr_ids, caption_ids, preprocess, tokenizer
    )
    trohn_images, trohn_tokens = trohn_batch_tensors(
        trohn_rows, preprocess, tokenizer
    )

    images = torch.cat([flickr_images, trohn_images], dim=0)
    tokens = torch.cat([flickr_tokens, trohn_tokens], dim=0)

    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        image_features = F.normalize(model.encode_image(images).float(), dim=-1)
        text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)
        image_features = adapt(image_features, image_adapter, alpha)
        text_features = adapt(text_features, text_adapter, alpha)

        fb = FLICKR_BATCH_SIZE
        tb = len(trohn_rows)
        flickr_loss = contrastive_loss(image_features[:fb], text_features[:fb])
        trohn_images_f = torch.cat([
            image_features[fb:fb + tb],
            image_features[fb + tb:fb + 2 * tb],
        ], dim=0)
        trohn_texts_f = torch.cat([
            text_features[fb:fb + tb],
            text_features[fb + tb:fb + 2 * tb],
        ], dim=0)
        trohn_loss = contrastive_loss(trohn_images_f, trohn_texts_f)
        loss = flickr_loss + TROHN_WEIGHT * trohn_loss

    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite training loss; stopping before optimizer update")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for _, p in trainable], GRAD_CLIP_NORM
    )
    scaler.step(optimizer)
    scaler.update()

    return (
        float(loss.detach().item()),
        float(flickr_loss.detach().item()),
        float(trohn_loss.detach().item()),
        float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
    )


def main():
    parser = argparse.ArgumentParser(
        description="E3-A partial end-to-end CLIP fine-tuning; safe check-only by default."
    )
    parser.add_argument("--train", action="store_true", help="Actually train; default is check-only.")
    parser.add_argument("--resume", action="store_true", help="Resume this E3 run from latest.pt.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Total target epochs for this E3 run; can be increased on resume.")
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS,
                        help="TROHN rows per epoch for the initial pilot (default: 10000).")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="CLIP trainable-parameter learning rate (default: 1e-6).")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.train_rows < TROHN_BATCH_SIZE or args.train_rows > EXPECTED_TRAIN_ROWS:
        parser.error(f"--train-rows must be between {TROHN_BATCH_SIZE} and {EXPECTED_TRAIN_ROWS}")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.resume and not args.train:
        parser.error("Use --train --resume to resume an existing E3 run")

    set_seed(SEED)
    context = preflight()

    model = context["model"]
    trainable = context["trainable"]
    image_adapter = context["image_adapter"]
    text_adapter = context["text_adapter"]
    alpha = context["alpha"]
    preprocess = context["preprocess"]
    tokenizer = context["tokenizer"]
    flickr_train = context["flickr_train"]
    flickr_dev = context["flickr_dev"]
    clean_indices = context["clean_indices"]
    manifest = context["manifest"]
    source_epoch = context["source_epoch"]

    if not args.train:
        print("\nCHECK-ONLY COMPLETE. No model weights, optimizer state, or run files were modified.")
        return

    script_hash = sha256_file(Path(__file__).resolve())
    source_hash = sha256_file(SOURCE_E2_CHECKPOINT)
    manifest_hash = data_signature(manifest)
    config = build_config(args, manifest_hash)

    latest_path = RUN_DIR / "latest.pt"
    history_path = RUN_DIR / "history.csv"
    summary_path = RUN_DIR / "metrics_summary.json"

    optimizer = torch.optim.AdamW(
        [p for _, p in trainable],
        lr=float(args.lr),
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    sampler = FlickrBatchSampler(len(flickr_train), seed=SEED + 10_000)

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"Cannot resume; missing {latest_path}")
        checkpoint = load_pt(latest_path)
        validate_resume_checkpoint(
            checkpoint, config, source_hash, script_hash, manifest_hash
        )
        load_trainable_state_dict(model, checkpoint["trainable_model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        sampler.load_state_dict(checkpoint["flickr_sampler_state"])
        restore_rng_state(checkpoint)

        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        else:
            # Recovery for the narrow case where the process stopped after the
            # initial latest.pt was written but before the baseline summary.
            if int(checkpoint["global_step"]) != 0 or int(checkpoint["step_in_epoch"]) != 0:
                raise FileNotFoundError(
                    f"Missing E3 model-selection summary after training had started: {summary_path}"
                )
            print("[RECOVERY] Baseline summary missing; recomputing the E2 baseline.", flush=True)
            baseline_metrics = evaluate_all(
                model, image_adapter, text_adapter, flickr_dev, clean_indices,
                preprocess, tokenizer, alpha, "E2 baseline recovery",
            )
            verify_baseline_matches_e2(baseline_metrics, context["e2_checkpoint"])
            summary = initialize_best_checkpoints(
                model, image_adapter, text_adapter, alpha, baseline_metrics,
                source_epoch, source_hash, script_hash, summary_path, config,
            )

        completed_epochs = int(checkpoint["completed_epochs"])
        global_step = int(checkpoint["global_step"])
        if checkpoint["status"] == "in_progress":
            start_epoch = int(checkpoint["epoch_in_progress"])
            resume_step = int(checkpoint["step_in_epoch"])
            resume_rows_seen = int(checkpoint["rows_seen_in_epoch"])
            resume_total_loss = float(checkpoint["total_loss"])
            resume_total_flickr = float(checkpoint["total_flickr"])
            resume_total_trohn = float(checkpoint["total_trohn"])
        elif checkpoint["status"] == "epoch_complete":
            start_epoch = completed_epochs + 1
            resume_step = 0
            resume_rows_seen = 0
            resume_total_loss = 0.0
            resume_total_flickr = 0.0
            resume_total_trohn = 0.0
        else:
            raise RuntimeError(f"Unknown checkpoint status: {checkpoint['status']}")

        print("[RESUME] completed epochs:", completed_epochs)
        print("[RESUME] next epoch:", start_epoch)
        print("[RESUME] step in epoch:", resume_step)
        print("[RESUME] TROHN rows already consumed:", resume_rows_seen)
    else:
        if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
            raise FileExistsError(
                f"{RUN_DIR} already contains an E3 run. Use --train --resume; "
                "do not overwrite previous experiments."
            )
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        completed_epochs = 0
        global_step = 0
        start_epoch = 1
        resume_step = 0
        resume_rows_seen = 0
        resume_total_loss = 0.0
        resume_total_flickr = 0.0
        resume_total_trohn = 0.0

        # Save an initial recoverable state BEFORE the expensive baseline pass.
        save_latest(
            latest_path, model, optimizer, scaler, sampler,
            epoch=1, step_in_epoch=0, rows_seen=0, completed_epochs=0,
            global_step=0, total_loss=0.0, total_flickr=0.0, total_trohn=0.0,
            status="in_progress", config=config, source_hash=source_hash,
            script_hash=script_hash, manifest_hash=manifest_hash,
        )

        print("\n=== E2 baseline re-evaluation on raw clean validation ===")
        baseline_metrics = evaluate_all(
            model, image_adapter, text_adapter, flickr_dev, clean_indices,
            preprocess, tokenizer, alpha, "E2 baseline",
        )
        verify_baseline_matches_e2(baseline_metrics, context["e2_checkpoint"])
        summary = initialize_best_checkpoints(
            model, image_adapter, text_adapter, alpha,
            baseline_metrics, source_epoch, source_hash,
            script_hash, summary_path, config,
        )

    if start_epoch > args.epochs:
        print("Requested target epochs already completed; no training needed.")
        print("Best TROHN:", RUN_DIR / "best_trohn_group.pt")
        print("Best Flickr:", RUN_DIR / "best_flickr.pt")
        return

    steps_per_epoch = math.ceil(args.train_rows / TROHN_BATCH_SIZE)
    print("\nE3-A training configuration")
    print("Run directory:", RUN_DIR)
    print("Epoch target:", args.epochs)
    print("TROHN rows/epoch:", args.train_rows)
    print("Steps/epoch:", steps_per_epoch)
    print("TROHN/Flickr batch:", TROHN_BATCH_SIZE, "/", FLICKR_BATCH_SIZE)
    print("CLIP learning rate:", args.lr)
    print("Frozen E2 Adapter alpha:", alpha)
    print("Checkpoint interval:", SAVE_EVERY_STEPS, "steps")

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == start_epoch:
            step_start = resume_step
            rows_seen = resume_rows_seen
            total_loss = resume_total_loss
            total_flickr = resume_total_flickr
            total_trohn = resume_total_trohn
        else:
            step_start = 0
            rows_seen = 0
            total_loss = 0.0
            total_flickr = 0.0
            total_trohn = 0.0

        if rows_seen > args.train_rows:
            raise RuntimeError("Resume checkpoint rows_seen exceeds train_rows")
        if rows_seen != step_start * TROHN_BATCH_SIZE:
            # Every step except a possible final partial batch is full-sized.
            expected_rows_seen = min(step_start * TROHN_BATCH_SIZE, args.train_rows)
            if rows_seen != expected_rows_seen:
                raise RuntimeError(
                    f"Resume cursor mismatch: rows_seen={rows_seen}, "
                    f"expected={expected_rows_seen}"
                )

        print(f"\nEpoch {epoch}/{args.epochs} | resuming at step {step_start}/{steps_per_epoch}", flush=True)
        trohn_dataset = load_trohn_stream("train", shuffle_seed=SEED + epoch)
        if rows_seen:
            trohn_dataset = trohn_dataset.skip(rows_seen)
        trohn_iterator = iter(trohn_dataset)

        for step_index in range(step_start, steps_per_epoch):
            batch_size = min(TROHN_BATCH_SIZE, args.train_rows - rows_seen)
            trohn_rows = collect_rows(trohn_iterator, batch_size)
            if len(trohn_rows) != batch_size:
                raise RuntimeError(
                    f"TROHN stream ended early: got {len(trohn_rows)}, expected {batch_size}"
                )

            loss_value, flickr_value, trohn_value, grad_norm = train_one_step(
                model, trainable, image_adapter, text_adapter, alpha,
                optimizer, scaler, flickr_train, sampler, trohn_rows,
                preprocess, tokenizer,
            )

            rows_seen += len(trohn_rows)
            global_step += 1
            completed_step = step_index + 1
            total_loss += loss_value
            total_flickr += flickr_value
            total_trohn += trohn_value

            if completed_step % 50 == 0 or completed_step == steps_per_epoch:
                print(
                    f"Epoch {epoch}/{args.epochs} | Step {completed_step}/{steps_per_epoch} | "
                    f"Loss={total_loss / completed_step:.5f} | "
                    f"Flickr={total_flickr / completed_step:.5f} | "
                    f"TROHN={total_trohn / completed_step:.5f} | "
                    f"GradNorm={grad_norm:.4f}",
                    flush=True,
                )

            if global_step % SAVE_EVERY_STEPS == 0:
                save_latest(
                    latest_path, model, optimizer, scaler, sampler,
                    epoch=epoch, step_in_epoch=completed_step,
                    rows_seen=rows_seen, completed_epochs=completed_epochs,
                    global_step=global_step, total_loss=total_loss,
                    total_flickr=total_flickr, total_trohn=total_trohn,
                    status="in_progress", config=config,
                    source_hash=source_hash, script_hash=script_hash,
                    manifest_hash=manifest_hash,
                )
                print("[SAVED STEP CHECKPOINT]", latest_path, flush=True)

        if rows_seen != args.train_rows:
            raise RuntimeError(
                f"Epoch consumed {rows_seen} TROHN rows; expected {args.train_rows}"
            )

        print(f"\n=== E3-A epoch {epoch} clean validation ===", flush=True)
        metrics = evaluate_all(
            model, image_adapter, text_adapter, flickr_dev, clean_indices,
            preprocess, tokenizer, alpha, f"E3-A epoch {epoch}",
        )
        mean_loss = total_loss / steps_per_epoch
        mean_flickr = total_flickr / steps_per_epoch
        mean_trohn = total_trohn / steps_per_epoch

        update_best_checkpoints(
            model, image_adapter, text_adapter, alpha, metrics, epoch,
            summary, summary_path, source_hash, script_hash,
        )
        history_row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": mean_loss,
            "flickr_loss": mean_flickr,
            "trohn_loss": mean_trohn,
            **metrics,
            "learning_rate": args.lr,
            "train_rows": rows_seen,
        }
        write_history_row(history_path, history_row)

        epoch_payload = build_latest_payload(
            model, optimizer, scaler, sampler,
            epoch=epoch, step_in_epoch=steps_per_epoch,
            rows_seen=rows_seen, completed_epochs=epoch,
            global_step=global_step, total_loss=mean_loss,
            total_flickr=mean_flickr, total_trohn=mean_trohn,
            status="epoch_complete", config=config,
            source_hash=source_hash, script_hash=script_hash,
            manifest_hash=manifest_hash,
        )
        atomic_torch_save(epoch_payload, latest_path)
        atomic_torch_save(epoch_payload, RUN_DIR / f"epoch_{epoch:02d}.pt")
        print("[SAVED EPOCH CHECKPOINT]", RUN_DIR / f"epoch_{epoch:02d}.pt", flush=True)

        # Clear in-progress resume cursor for the next epoch.
        resume_step = 0
        resume_rows_seen = 0
        resume_total_loss = 0.0
        resume_total_flickr = 0.0
        resume_total_trohn = 0.0
        completed_epochs = epoch

    print("\n" + "=" * 76)
    print("E3-A TRAINING COMPLETE")
    print("History:", history_path)
    print("Latest:", latest_path)
    print("Best TROHN:", RUN_DIR / "best_trohn_group.pt")
    print("Best Flickr:", RUN_DIR / "best_flickr.pt")
    print("Summary:", summary_path)
    print("=" * 76)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()