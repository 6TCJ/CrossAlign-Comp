from pathlib import Path
import json

ROOT = Path.cwd()

if not (ROOT / "train_e2_matched_clean.py").exists():
    raise RuntimeError("当前目录不是 CrossAlign 项目根目录")

checks = [
    ("COCO full 原始图片",
     ROOT / "e2_data/coco_full/train2017", "*.jpg"),

    ("COCO pilot 原始图片",
     ROOT / "e2_data/coco_images", "*.jpg"),

    ("TROHN 负样本图片",
     ROOT / "e2_data/trohn_negative_images", "*.jpg"),

    ("TROHN 训练特征",
     ROOT / "e2_data/e2_b_full/train", "*.pt"),

    ("TROHN 验证特征",
     ROOT / "e2_data/e2_b_full/validation", "*.pt"),
]

print("\n========== E3 DATA AUDIT ==========")

for name, path, pattern in checks:
    if path.exists():
        count = sum(1 for _ in path.glob(pattern))
        print(f"[FOUND] {name}: {count}")
        print(" ", path)
    else:
        print(f"[MISSING] {name}")
        print(" ", path)

print("\n========== CHECKPOINT AUDIT ==========")

files = [
    ROOT / "clip_train_features.pt",
    ROOT / "clip_dev_features.pt",
    ROOT / "e2_data/continuation_full_text_image_v2/best_trohn_group.pt",
    ROOT / "e2_data/continuation_full_text_image_v2/best_flickr.pt",
    ROOT / "e2_data/continuation_full_text_image_v2/latest.pt",
]

for p in files:
    print(f"[{'FOUND' if p.exists() else 'MISSING'}] {p}")

summary = ROOT / "e2_data/continuation_full_text_image_v2/metrics_summary.json"

if summary.exists():
    with summary.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n========== TRAINING SUMMARY ==========")
    print(json.dumps(data, indent=2, ensure_ascii=False))

print("\n========== AUDIT COMPLETE ==========")
