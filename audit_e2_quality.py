
import csv
import random
import re
import html
import base64
from pathlib import Path

random.seed(42)

data_dir = Path("e2_data")
input_csv = data_dir / "trohn_clip_scored.csv"
output_html = data_dir / "qc_review_24.html"

with open(input_csv, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# 分成三个难度组
groups = {
    "MODEL_PREFERS_NEGATIVE": [],
    "CLOSE_SCORES": [],
    "MODEL_DISTINGUISHES": []
}

for row in rows:
    groups[row["difficulty_signal"]].append(row)

# 每组随机抽8条，防止只看到极端样本
selected = []

for group_name, group_rows in groups.items():
    sample = random.sample(
        group_rows, min(8, len(group_rows))
    )
    selected.extend(sample)

# 生成HTML图文审核页面
parts = [
    "<html><head><meta charset='utf-8'>",
    "<style>",
    "body{font-family:Arial,sans-serif;background:#f5f6f8;"
    "margin:32px;max-width:1100px}",
    ".card{background:white;padding:20px;margin:20px 0;"
    "border-radius:12px;border:1px solid #ddd}",
    "img{max-width:380px;max-height:280px;object-fit:contain}",
    ".pos{color:#16734a}.neg{color:#b42318}",
    "</style></head><body>",
    "<h1>CrossAlign E2 | Data Quality Audit</h1>",
    "<p>Each sample: check whether the positive caption "
    "matches the image and the negative caption is truly false.</p>"
]

for index, row in enumerate(selected, 1):
    path = Path(row["image_path"])

    if not path.exists():
        path = data_dir / "coco_images" / row["image_id"]

    if path.exists():
        encoded = base64.b64encode(
            path.read_bytes()
        ).decode("ascii")
        img_html = (
            f'<img src="data:image/jpeg;base64,{encoded}">'
        )
    else:
        img_html = "<p>Image not found</p>"

    parts.append(
        f"<div class='card'>"
        f"<h2>Sample {index} | {html.escape(row['image_id'])}</h2>"
        f"<p>Group: {html.escape(row['difficulty_signal'])}"
        f" | Margin: {html.escape(row['clip_margin'])}"
        f" | Type: {html.escape(row['candidate_type'])}</p>"
        f"{img_html}"
        f"<p class='pos'><b>Positive:</b> "
        f"{html.escape(row['positive'])}</p>"
        f"<p class='neg'><b>Negative:</b> "
        f"{html.escape(row['negative'])}</p>"
        f"</div>"
    )

parts.append("</body></html>")

output_html.write_text(
    "\n".join(parts), encoding="utf-8"
)

print("Total scored:", len(rows))
print("Audit samples:", len(selected))

for name, items in groups.items():
    print(name, len(items))

print("Audit page:", output_html.resolve())
print("Open this HTML file in your browser.")
