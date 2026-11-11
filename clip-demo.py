
import torch
import open_clip
from PIL import Image

# 1. 使用RTX 5060
device = "cuda"
print("Device:", torch.cuda.get_device_name(0))

# 2. 加载预训练CLIP模型
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model = model.to(device).eval()

# 3. 处理图片
image = Image.open("test.jpg").convert("RGB")
image = preprocess(image).unsqueeze(0).to(device)

# 4. 准备文字描述
captions = [
    "A dog running on the grass",
    "A dog is being held by someone",
    "A car driving on the road",
    "A person riding a bicycle",
]

tokenizer = open_clip.get_tokenizer("ViT-B-32")
text = tokenizer(captions).to(device)

# 5. 图片和文字分别经过编码器
with torch.inference_mode():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # 将向量归一化
    image_features = torch.nn.functional.normalize(
        image_features, dim=-1
    )
    text_features = torch.nn.functional.normalize(
        text_features, dim=-1
    )

    # 计算图文余弦相似度
    scores = (image_features @ text_features.T)[0]

# 6. 显示结果
print("Image embedding:", image_features.shape)
print("Text embeddings:", text_features.shape)

print("\nMatching results:")
for caption, score in sorted(
    zip(captions, scores.tolist()),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{score:.4f}  {caption}")
