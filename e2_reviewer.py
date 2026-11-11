"""CrossAlign-Comp E2 hard-negative image-text reviewer.

Run from the CrossAlign project root:
    python -m pip install streamlit
    python -m streamlit run e2_reviewer.py

This script NEVER overwrites the original candidates.csv.
"""

import csv
import io
import os
from pathlib import Path

import streamlit as st
from datasets import DownloadConfig, load_dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "hard_negative_candidates"
SOURCE_CSV = DATA_DIR / "candidates.csv"
REVIEWED_CSV = DATA_DIR / "reviewed_candidates.csv"
FIELDS = ["image_id", "positive", "negative", "edit", "status", "review_note"]

# Review of the first 12 image/caption pairs. These are only preliminary
# visual checks; all remaining candidates must be reviewed before training.
SEED_REVIEWS = {
    0: ("ACCEPT", "Pink dress; purple is incorrect."),
    1: ("ACCEPT", "Black dog; brown substitution is incorrect."),
    4: ("ACCEPT", "Orange hat; typo corrected in local training copy."),
    5: ("ACCEPT", "Red ropes; gray substitution is incorrect."),
    6: ("ACCEPT", "White fence; gray substitution is incorrect."),
    7: ("ACCEPT", "White dog; gray substitution is incorrect."),
    9: ("UNCERTAIN", "Dark mottled dog; avoid ambiguous coat-color label."),
    10: ("ACCEPT", "Brown markings, not gray."),
    13: ("ACCEPT", "Brown dog, not gray."),
    14: ("ACCEPT", "White dog, not black."),
    15: ("ACCEPT", "Black and white coat, not brown."),
    17: ("ACCEPT", "Black dog in surf, not gray."),
}


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def prepare_review_file():
    if REVIEWED_CSV.exists():
        return read_rows(REVIEWED_CSV)
    if not SOURCE_CSV.exists():
        st.error(f"未找到原始CSV：{SOURCE_CSV}。请从 crossalign-comp 项目根目录运行本脚本。")
        st.stop()
    rows = read_rows(SOURCE_CSV)
    for row in rows:
        row.setdefault("status", "UNVERIFIED")
        row.setdefault("review_note", "")
        image_id = int(row["image_id"])
        if image_id in SEED_REVIEWS:
            row["status"], row["review_note"] = SEED_REVIEWS[image_id]
            if image_id == 4:
                # Fix typo only in the generated, reviewed CSV; never edit Flickr8k.
                row["positive"] = row["positive"].replace("starring at", "staring at")
                row["negative"] = row["negative"].replace("starring at", "staring at")
    write_rows(REVIEWED_CSV, rows)
    return rows


@st.cache_resource(show_spinner="首次加载本地Flickr8k图片缓存……")
def load_images():
    return load_dataset(
        "intro/flickr8k",
        data_files={"train": "train/**"},
        split="train",
        download_config=DownloadConfig(max_retries=10),
    )


st.set_page_config(page_title="CrossAlign E2 负样本审核", layout="wide")
st.title("CrossAlign · E2 负样本审核")
st.caption("左侧看原图，右侧看正负描述。审核结果自动另存为 reviewed_candidates.csv，不覆盖原始文件。")
rows = prepare_review_file()
if not rows:
    st.error("候选CSV中没有数据。")
    st.stop()

if "pos" not in st.session_state:
    st.session_state.pos = 0

counts = {k: sum(row["status"] == k for row in rows) for k in ["ACCEPT", "REJECT", "UNCERTAIN", "UNVERIFIED"]}
st.progress((len(rows) - counts["UNVERIFIED"]) / len(rows))
st.write(
    f"**已审核 {len(rows) - counts['UNVERIFIED']} / {len(rows)}**　"
    f"✅ 可用 {counts['ACCEPT']}　❌ 拒绝 {counts['REJECT']}　"
    f"❔ 不确定 {counts['UNCERTAIN']}"
)


def next_unverified(start):
    for step in range(1, len(rows) + 1):
        index = (start + step) % len(rows)
        if rows[index]["status"] == "UNVERIFIED":
            return index
    return min(start + 1, len(rows) - 1)


nav_left, nav_mid, nav_right = st.columns([1, 1, 3])
with nav_left:
    if st.button("← 上一条", use_container_width=True):
        st.session_state.pos = (st.session_state.pos - 1) % len(rows)
        st.rerun()
with nav_mid:
    if st.button("下一条 →", use_container_width=True):
        st.session_state.pos = (st.session_state.pos + 1) % len(rows)
        st.rerun()
with nav_right:
    if st.button("跳到下一条未审核样本", use_container_width=True, type="primary"):
        st.session_state.pos = next_unverified(st.session_state.pos)
        st.rerun()

pos = st.session_state.pos
row = rows[pos]
image_id = int(row["image_id"])
st.subheader(f"第 {pos + 1} / {len(rows)} 条　|　Image ID = {image_id}　|　当前：{row['status']}")

left, right = st.columns([1.1, 1], gap="large")
with left:
    images = load_images()
    if not 0 <= image_id < len(images):
        st.error(f"Image ID {image_id} 超出Flickr8k Train范围。")
        st.stop()
    st.image(images[image_id]["image"], caption=f"Flickr8k Train · Image ID {image_id}", use_container_width=True)
with right:
    st.markdown("**正确描述（Positive）**")
    pos_text = st.text_area("Positive", value=row["positive"], height=115, label_visibility="collapsed", key=f"pos_{image_id}")
    st.markdown("**候选错误描述（Negative）**")
    neg_text = st.text_area("Negative", value=row["negative"], height=115, label_visibility="collapsed", key=f"neg_{image_id}")
    st.caption("修改方式：" + row.get("edit", ""))
    note = st.text_input("审核说明（可选）", value=row.get("review_note", ""), key=f"note_{image_id}")
    st.info("ACCEPT：两句语法合理，Positive符合图片，Negative明确不符合图片；有疑问就标UNCERTAIN。")

    def decide(status):
        if status == "ACCEPT" and (not pos_text.strip() or not neg_text.strip() or pos_text.strip() == neg_text.strip()):
            st.error("正负描述不能为空或完全相同。")
            return
        rows[pos]["positive"] = pos_text.strip()
        rows[pos]["negative"] = neg_text.strip()
        rows[pos]["review_note"] = note
        rows[pos]["status"] = status
        write_rows(REVIEWED_CSV, rows)
        st.session_state.pos = next_unverified(pos)
        st.rerun()

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("✅ 通过", use_container_width=True):
            decide("ACCEPT")
    with b2:
        if st.button("❌ 拒绝", use_container_width=True):
            decide("REJECT")
    with b3:
        if st.button("❔ 不确定", use_container_width=True):
            decide("UNCERTAIN")

st.divider()
st.caption(f"自动保存路径：{REVIEWED_CSV}")

accepted = [r for r in rows if r["status"] == "ACCEPT"]
if accepted:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(accepted)
    st.download_button(
        f"导出已审核通过数据（{len(accepted)}条）",
        buf.getvalue().encode("utf-8-sig"),
        file_name="e2_accepted_hard_negatives.csv",
        mime="text/csv",
    )
