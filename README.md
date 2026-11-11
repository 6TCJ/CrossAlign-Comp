# CrossAlign-Comp

**Fine-Grained Image-Text Retrieval Optimization with OpenCLIP**

CrossAlign-Comp is a multimodal learning project focused on improving
fine-grained image-text matching and cross-modal retrieval.

The project progressively explores parameter-efficient adaptation,
hard-negative learning, paired image-text supervision, and partial
fine-tuning of pretrained CLIP representations.

## 1. Base Model

- Architecture: OpenCLIP ViT-B/32
- Pretrained weights: laion2b_s34b_b79k
- Embedding dimension: 512
- Framework: PyTorch
- Evaluation: Flickr8k Dev and TROHN-Img CLEAN Validation

## 2. Methodology

### E0: CLIP Baseline

Establish the original image-to-text (I2T) and text-to-image (T2I)
retrieval performance using pretrained OpenCLIP.

### E1: Dual-Side Adapter

Introduce separate lightweight MLP adapters for image and text
representations.

Residual adaptation:

    f_out = Normalize(f + alpha * Adapter(f))

### E2-A: Text-Level Hard Negative Learning

Explore fine-grained textual attribute discrimination through
hard-negative text pairs and controlled experiments.

### E2-B: Paired Image-Text Hard Negative Learning

Introduce TROHN-Img to extend hard-negative supervision from
text-only discrimination to paired image-text matching.

Compare:
- Full control
- Text-only hard-negative learning
- Paired image-text hard-negative learning

### Data Quality Audit

An image_id overlap audit identified leakage between the original
TROHN-Img training and validation partitions.

A CLEAN Validation split was constructed by excluding overlapping
image IDs.

- CLEAN validation rows: 5,788
- Unique validation images: 4,715
- Train/validation image overlap: 0

All subsequent reported TROHN validation comparisons use the
cleaned validation protocol.

### E3: Partial CLIP Fine-Tuning

Move beyond cached-feature Adapter-only training.

- Online image and text encoding
- Partial visual Transformer fine-tuning
- Partial text Transformer fine-tuning
- Projection and normalization parameter adaptation
- Frozen E2 adapters
- Hard-negative supervision retained

Trainable parameters: 10,898,176 / 151,277,313 (approximately 7.2%).

## 3. Experimental Results

E3 results on the project evaluation protocol:

| Metric | E2 Baseline | E3 Epoch 12 |
|---|---:|---:|
| Flickr I2T@1 | 77.50% | 76.40% |
| Flickr T2I@1 | 62.06% | 63.34% |
| TROHN I2T | 70.66% | 81.93% |
| TROHN T2I | 50.43% | 64.03% |
| TROHN Group | 46.67% | 61.66% |

E3 model selection:

- Best TROHN Group: Epoch 12, 61.66%
- Best Flickr retrieval: Epoch 6, I2T 78.10%, T2I 64.56%

The two checkpoints are selected independently according to
their respective evaluation objectives.

## 4. Engineering Features

- Cached-feature training pipeline for Adapter experiments
- Online image-text encoding for partial CLIP fine-tuning
- Gradient smoke tests
- Image-level data leakage auditing
- Matched-condition ablation experiments
- Step-level checkpointing
- Optimizer and AMP scaler state restoration
- RNG and data sampler state restoration
- Independent best-model selection
- CSV and JSON experiment tracking

## 5. Project Structure

The repository contains scripts for:

- Dataset inspection and preprocessing
- Feature extraction and caching
- Hard-negative construction
- Adapter training and evaluation
- Matched ablation experiments
- Partial CLIP fine-tuning
- Data quality auditing
- Checkpoint analysis and experiment reporting

## 6. Environment

The experiments were conducted using:

- Python 3.10
- PyTorch 2.5.1
- CUDA 12.4
- OpenCLIP 2.26.1
- NVIDIA GeForce RTX 3090

See `requirements_server.txt` for the recorded server environment.

## 7. Data and Model Weights

Training datasets, cached features, and model checkpoints are not
included in this repository.

Users must obtain the corresponding datasets and pretrained
weights separately and configure their local data paths.

## 8. Project Status

This repository documents an experimental research and engineering
pipeline. Reported results correspond to the recorded experimental
checkpoints and CLEAN Validation protocol.

The project is intended for research, learning, and reproducibility
purposes rather than as a production-ready deployment package.
