
from datasets import load_dataset

dataset = load_dataset(
    "intro/flickr8k",
    data_files={"train": "train/**"},
    split="train"
)

print("Total training images:", len(dataset))

for i in range(3):
    print(f"\n=== Training Image {i} ===")

    for j in range(5):
        print(
            f"Caption {j}:",
            dataset[i][f"caption_{j}"]
        )
