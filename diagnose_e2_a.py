
import csv
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "e2_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.05



class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def load_pt(path):
    return torch.load(
        path, map_location="cpu", weights_only=True
    )


def build(path):
    ckpt = load_pt(path)
    modules = []

    for key in ("image_adapter", "text_adapter"):
        state = ckpt[key]

        module = Adapter(
            state["fc1.weight"].shape[1],
            state["fc1.weight"].shape[0]
        )

        module.load_state_dict(state, strict=True)
        modules.append(module.to(DEVICE).eval())

    return modules


def adapt(x, module):
    return F.normalize(
        x + ALPHA * module(x), dim=-1
    )


hard = load_pt(DATA / "e2_attr_features.pt")
dev = load_pt(ROOT / "clip_dev_features.pt")

idx_train = set(hard["train_indices"].tolist())
idx_val = set(hard["val_indices"].tolist())

assert idx_train.isdisjoint(idx_val)

n = len(hard["image_ids"])
assert len(idx_train) + len(idx_val) == n

with (
    DATA / "e2_binding_scored.csv"
).open(encoding="utf-8-sig", newline="") as f:
    meta = {
        r["image_id"]: r
        for r in csv.DictReader(f)
    }


models = {
    "E1": ROOT / "clip_adapter_e1.pt",
    "Control": DATA / "e2_a_control.pt",
    "E2A": DATA / "e2_a_hard.pt"
}

gaps = {}
retrieval = {}

with torch.inference_mode():
    for name, path in models.items():

        ia, ta = build(path)

        # 属性绑定训练/验证样本
        i = adapt(
            hard["image_features"].to(DEVICE).float(),
            ia
        )
        p = adapt(
            hard["positive_features"].to(DEVICE).float(),
            ta
        )
        neg = adapt(
            hard["negative_features"].to(DEVICE).float(),
            ta
        )

        gaps[name] = (
            (i * p).sum(-1) - (i * neg).sum(-1)
        ).cpu()

        # Flickr8k Dev 检索
        di = adapt(
            dev["image_features"].to(DEVICE).float(),
            ia
        )
        dt = adapt(
            dev["text_features"].to(DEVICE).float(),
            ta
        )

        similarity = di @ dt.T

        query_i = torch.arange(
            len(di), device=DEVICE
        )
        query_t = torch.arange(
            len(dt), device=DEVICE
        ) // 5

        output = {}

        for k in (1, 5, 10):
            it = similarity.topk(
                k, dim=1
            ).indices // 5

            ti = similarity.T.topk(
                k, dim=1
            ).indices

            output[f"I2T@{k}"] = (
                (it == query_i[:, None])
                .any(1).float().mean().item()
            )

            output[f"T2I@{k}"] = (
                (ti == query_t[:, None])
                .any(1).float().mean().item()
            )

        retrieval[name] = output


print("\n=== Flickr8k Dev ===")

for name, values in retrieval.items():
    print(
        name,
        " | ".join(
            f"{metric}: {value:.2%}"
            for metric, value in values.items()
        )
    )


print("\n=== Hard Candidate Train/Val ===")

for name, gap in gaps.items():
    for split, indices in (
        ("train", sorted(idx_train)),
        ("val", sorted(idx_val))
    ):
        x = gap[indices]

        print(
            f"{name:8s} {split:5s}: "
            f"{(x > 0).sum().item()}/{len(x)} "
            f"= {(x > 0).float().mean().item():.2%}, "
            f"mean gap={x.mean().item():+.5f}"
        )


# 分析Control和E2-A的逐题变化
val_idx = sorted(idx_val)

base_correct = gaps["Control"][val_idx] > 0
hard_correct = gaps["E2A"][val_idx] > 0

fixed = (
    ~base_correct & hard_correct
).sum().item()

broken = (
    base_correct & ~hard_correct
).sum().item()

print("\n=== Paired E2-A vs Control ===")
print("E2-A fixed:", fixed)
print("E2-A broke:", broken)
print("Net gain:", fixed - broken)


# 保存213条样本的逐题分析
rows = []

for j, image_id in enumerate(hard["image_ids"]):
    m = meta.get(image_id, {})

    c = gaps["Control"][j].item()
    h = gaps["E2A"][j].item()

    rows.append({
        "image_id": image_id,
        "split": (
            "train" if j in idx_train else "val"
        ),
        "positive": m.get("positive", ""),
        "negative": m.get("negative", ""),
        "E0_margin": m.get("clip_margin", ""),
        "E1_gap": round(
            gaps["E1"][j].item(), 6
        ),
        "Control_gap": round(c, 6),
        "E2A_gap": round(h, 6),
        "Control_correct": c > 0,
        "E2A_correct": h > 0,
        "outcome": (
            "FIXED" if c <= 0 < h else
            "BROKEN" if h <= 0 < c else
            "UNCHANGED"
        )
    })


out = DATA / "e2_a_case_diagnostic.csv"

with out.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)


print("\n=== Val Cases With Decision Flips ===")

for row in rows:
    if (
        row["split"] == "val"
        and row["outcome"] != "UNCHANGED"
    ):
        print(
            row["outcome"],
            row["image_id"],
            f"Control={row['Control_gap']:+.5f}",
            f"E2A={row['E2A_gap']:+.5f}"
        )
        print("Positive:", row["positive"])
        print("Negative:", row["negative"])


print("\nSaved:", out)
