
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

OUTPUT = Path("e2_data/e2_binding_fullscan.csv")
OUTPUT.parent.mkdir(exist_ok=True)

# None表示扫描全部训练记录
MAX_SCAN = None

# 允许交换的属性
ATTRIBUTES = """
black white red blue green yellow pink purple orange
brown gray grey maroon beige turquoise gold golden
silver bronze chrome porcelain wooden metallic plastic
""".split()

# 明确的物体或物体部位
OBJECTS = """
dress dresses shirt shirts blouse blouses pants trousers
shorts skirt skirts jacket jackets coat coats
umbrella umbrellas hat hats helmet helmets
tie ties shoes socks boots gloves gear
car cars truck trucks bus buses bike bikes
bicycle bicycles motorcycle motorcycles
board boards surfboard surfboards boat boats
airplane airplanes plane planes
dog dogs cat cats bird birds cow cows
horse horses giraffe giraffes zebra zebras
bowl bowls tank tanks toilet toilets
cabinet cabinets floor floors wall walls curb curbs
hydrant hydrants fence fences bench benches
vase vases bottle bottles kite kites
phone phones frisbee frisbees ball balls
head heads beak beaks wing wings tail tails
towel towels bag bags purse purses
couch couches chair chairs table tables
sign signs building buildings house houses
desk desks door doors window windows
spoon spoons fork forks plate plates
banana bananas tree trees
""".split()

# 允许出现在属性与名词之间的修饰词
MODIFIERS = """
colored coloured painted tiled rain leather
wood racing fire cell phone tennis baseball
teddy kitchen
""".split()

def make_alternatives(words):
    return "|".join(
        sorted(
            [re.escape(w) for w in set(words)],
            key=len,
            reverse=True
        )
    )

attr_re = make_alternatives(ATTRIBUTES)
obj_re = make_alternatives(OBJECTS)
mod_re = make_alternatives(MODIFIERS)

pattern = re.compile(
    rf"\b({attr_re})\s+"
    rf"(?:(?:{mod_re})\s+){{0,2}}"
    rf"({obj_re})\b",
    re.IGNORECASE
)


def word_counter(text):
    return Counter(
        re.findall(r"[a-z0-9]+", text.lower())
    )


def extract_bindings(text):
    result = defaultdict(set)

    for match in pattern.finditer(text):
        attribute = match.group(1).lower()
        obj = match.group(2).lower()
        result[obj].add(attribute)

    return dict(result)


def detect_attribute_swap(positive, negative):
    # 词汇及其出现次数必须一致
    if word_counter(positive) != word_counter(negative):
        return None

    p = extract_bindings(positive)
    n = extract_bindings(negative)

    # 两句话必须识别出相同的物体集合
    if not p or set(p) != set(n):
        return None

    changed = [
        obj for obj in p
        if p[obj] != n[obj]
    ]

    # 本轮只保留两个对象发生属性交换
    if len(changed) != 2:
        return None

    a, b = sorted(changed)

    if not all(
        len(mapping[obj]) == 1
        for mapping in (p, n)
        for obj in (a, b)
    ):
        return None

    pa = next(iter(p[a]))
    pb = next(iter(p[b]))
    na = next(iter(n[a]))
    nb = next(iter(n[b]))

    # 必须真正互换两个不同属性
    if pa == pb:
        return None

    if not (pa == nb and pb == na):
        return None

    return {
        "object_a": a,
        "object_b": b,
        "attribute_a": pa,
        "attribute_b": pb,
    }


# 内置回归测试
assert detect_attribute_swap(
    "A woman in a yellow rain coat with a red umbrella.",
    "A woman in a red rain coat with a yellow umbrella."
) is not None

assert detect_attribute_swap(
    "A stir fry with shrimp and broccoli.",
    "A stir fry with broccoli and shrimp."
) is None

assert detect_attribute_swap(
    "A silver airplane in a blue sky.",
    "A blue airplane in a silver sky."
) is None

assert detect_attribute_swap(
    "A bunch of green bananas on a banana tree.",
    "A bunch of bananas on a green banana tree."
) is None

print("Loading TROHN-Img Train metadata...")

dataset = load_dataset(
    "imirandam/TROHN-Img",
    split="train",
    streaming=True,
    columns=[
        "image_id",
        "caption",
        "negative_caption"
    ]
)

fields = [
    "image_id",
    "positive",
    "negative",
    "candidate_type",
    "object_a",
    "object_b",
    "attribute_a",
    "attribute_b",
    "label_status"
]

scanned = 0
kept = 0
seen_images = set()
completed = False

# 一边扫描一边保存，网络中断时保留已发现的结果
with OUTPUT.open(
    "w", encoding="utf-8-sig", newline=""
) as f:

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    try:
        for row in dataset:
            if MAX_SCAN is not None and scanned >= MAX_SCAN:
                completed = True
                break

            scanned += 1

            image_id = str(row["image_id"])
            positive = str(row["caption"]).strip()
            negative = str(row["negative_caption"]).strip()

            if image_id in seen_images:
                continue

            result = detect_attribute_swap(
                positive, negative
            )

            if result is not None:
                writer.writerow({
                    "image_id": image_id,
                    "positive": positive,
                    "negative": negative,
                    "candidate_type": "ATTRIBUTE_SWAP",
                    **result,
                    "label_status": "UNVERIFIED"
                })

                kept += 1
                seen_images.add(image_id)
                f.flush()

            if scanned % 10000 == 0:
                print(
                    f"Scanned: {scanned} | "
                    f"Attribute swaps: {kept}"
                )

        else:
            completed = True

    except Exception as e:
        print("\nScan interrupted!")
        print("Error:", type(e).__name__, str(e)[:300])

print("\n=== E2 Full Scan Results ===")
print("Scanned:", scanned)
print("Attribute swap candidates:", kept)
print("Scan completed:", completed)
print("Saved:", OUTPUT.resolve())

with OUTPUT.open(
    encoding="utf-8-sig", newline=""
) as f:
    samples = list(csv.DictReader(f))

print("\n=== First 8 Candidates ===")
for row in samples[:8]:
    print("\nImage:", row["image_id"])
    print("Positive:", row["positive"])
    print("Negative:", row["negative"])
    print(
        "Swap:",
        row["attribute_a"], row["object_a"],
        "<->",
        row["attribute_b"], row["object_b"]
    )
