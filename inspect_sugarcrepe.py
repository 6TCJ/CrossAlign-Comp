
from datasets import load_dataset
from PIL import Image

print("Loading SugarCrepe...")

dataset = load_dataset(
    "HuggingFaceM4/SugarCrepe_swap_att",
    split="test"
)

print("\nNumber of samples:", len(dataset))
print("Columns:", dataset.column_names)

sample = dataset[0]

print("\n=== First Sample ===")

for key, value in sample.items():
    if isinstance(value, Image.Image):
        print(f"{key}: Image {value.size}")
        value.save("sugarcrepe_sample.jpg")
    elif isinstance(value, str):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {type(value).__name__}")

print("\n=== Real Captions ===")

print("Tested labels:", sample["tested_labels"])
print("True label:", sample["true_label"])

print("\n=== Second Sample ===")
print("Tested labels:", dataset[1]["tested_labels"])
print("True label:", dataset[1]["true_label"])