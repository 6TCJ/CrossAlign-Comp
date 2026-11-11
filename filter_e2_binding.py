
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("e2_data")

SOURCE = DATA / "trohn_train_candidates.csv"
SCORED = DATA / "trohn_clip_scored.csv"
OUTPUT = DATA / "e2_attribute_binding_candidates.csv"

# 颜色和材料属性
ATTRS = set("""
black white red blue green yellow pink purple orange
brown gray grey maroon beige turquoise gold golden
silver bronze chrome porcelain wooden metallic
plastic striped spotted
""".split())

# 防止把形容词、介词等错误识别成物体
IGNORE = set("""
and or of in on at to from with without before after
the a an is are was were small large little tiny big
tall short young old baby adult teddy
""".split())

pattern = re.compile(
    r"\b("
    + "|".join(sorted(ATTRS, key=len, reverse=True))
    + r")\s+(?:colou?red\s+)?([a-z]+)\b",
    re.IGNORECASE
)


def word_counter(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return Counter(words)


def extract_bindings(text):
    """提取属性与物体之间的绑定关系。"""

    result = defaultdict(Counter)

    for match in pattern.finditer(text):
        attribute = match.group(1).lower()
        noun = match.group(2).lower()

        if noun not in IGNORE:
            result[noun][attribute] += 1

    return dict(result)


def classify_pair(positive, negative):
    """严格筛选属性绑定发生变化的候选。"""

    # 两句话的词汇及其出现次数必须相同
    if word_counter(positive) != word_counter(negative):
        return None

    pos_bind = extract_bindings(positive)
    neg_bind = extract_bindings(negative)

    if not pos_bind or not neg_bind:
        return None

    # 至少涉及两个不同对象
    nouns = set(pos_bind) | set(neg_bind)

    if len(nouns) < 2:
        return None

    # 属性对应关系必须发生变化
    if pos_bind == neg_bind:
        return None

    # 属性本身不能被增加或删除
    pos_attrs = sum(pos_bind.values(), Counter())
    neg_attrs = sum(neg_bind.values(), Counter())

    if pos_attrs != neg_attrs:
        return None

    # 两个以上属性发生重新绑定，优先级更高
    if sum(pos_attrs.values()) >= 2:
        return "TWO_ATTRIBUTE_SWAP"

    return "ONE_ATTRIBUTE_MOVE"


# 测试筛选规则，防止明显错误
assert classify_pair(
    "A kitchen has red cabinets and yellow floor.",
    "A kitchen has yellow cabinets and red floor."
) == "TWO_ATTRIBUTE_SWAP"

assert classify_pair(
    "A man in a shirt and purple tie.",
    "A man in a purple shirt and tie."
) == "ONE_ATTRIBUTE_MOVE"

assert classify_pair(
    "Two white small teddy bears are sitting.",
    "Two small white teddy bears are sitting."
) is None

assert classify_pair(
    "A bathroom has a toilet, sink, and mirror.",
    "A bathroom has a sink, toilet, and mirror."
) is None


# 读取3000组候选
with open(SOURCE, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# 关联之前已经计算的200组CLIP分数
scored = {}

if SCORED.exists():
    with open(SCORED, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["image_id"],
                row["positive"],
                row["negative"]
            )
            scored[key] = row


results = []

for row in rows:
    category = classify_pair(
        row["positive"],
        row["negative"]
    )

    if category is None:
        continue

    key = (
        row["image_id"],
        row["positive"],
        row["negative"]
    )

    score = scored.get(key, {})

    results.append({
        "image_id": row["image_id"],
        "positive": row["positive"],
        "negative": row["negative"],
        "binding_type": category,
        "clip_margin": score.get("clip_margin", ""),
        "image_path": score.get("image_path", ""),
        "label_status": "CANDIDATE_NOT_VERIFIED"
    })


# 优先保留双属性交换，并展示已经计算Margin的记录
results.sort(
    key=lambda r: (
        r["binding_type"] != "TWO_ATTRIBUTE_SWAP",
        r["clip_margin"] == "",
        abs(float(r["clip_margin"]))
        if r["clip_margin"] else 0
    )
)

fields = [
    "image_id",
    "positive",
    "negative",
    "binding_type",
    "clip_margin",
    "image_path",
    "label_status"
]

with open(
    OUTPUT,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)


print("\n=== E2 Attribute Binding Selection ===")

print("Original candidates:", len(rows))
print("Binding candidates:", len(results))

print(
    "Two-attribute swaps:",
    sum(r["binding_type"] == "TWO_ATTRIBUTE_SWAP"
        for r in results)
)

print(
    "One-attribute moves:",
    sum(r["binding_type"] == "ONE_ATTRIBUTE_MOVE"
        for r in results)
)

print(
    "Already CLIP-scored:",
    sum(bool(r["clip_margin"]) for r in results)
)

print("\n=== First 8 Candidates ===")

for row in results[:8]:
    print("\nImage:", row["image_id"])
    print("Type:", row["binding_type"])
    print("Positive:", row["positive"])
    print("Negative:", row["negative"])
    print("Margin:", row["clip_margin"])

print("\nSaved:", OUTPUT)
