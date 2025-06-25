# CVGSSL

## The paper "CLIP-Vision Guided Few-Shot Metal Surface Defect Recognition" has been published in IEEE Transactions on Industrial Informatics.


## 🧠 CVGSSL Project  
Metal surface defect recognition (MSDR) based on deep learning encounters the challenge of Few-Shot expert-labeled data. In this study, we proposed a CLIP-Vision Guided Self Supervised Learning (CVGSSL) framework for representation learning of unlabeled data, completing MSDR using Few-Shot labeled data. This framework initially generates rich and diverse representation information through multiple CLIP-Vs to ensure effective SSL pretraining, followed by the design of an MLP-Adapter to distill knowledge and adapt these representations to recognition tasks. Additionally, we constructed a self-constrained loss to address the inherent problem of intra-class and interclass distance ambiguity that causes the representation to fall into an equivocal decision margin. Following labelfree pre-training of CVGSSL, the downstream model adapts to 1-shot to 4-shot defect recognition tasks through finetuning.

### 🚀 Getting Started

#### 1. Install Dependencies

```bash
pip install torch torchvision timm numpy
```

#### 2. Run the Training Script

```bash
python train.py 
```
#### 3. Run the Finetune Script

```bash
python finetune_lincls.py
```

### 📊 Logging & Output

Training logs are automatically saved under `logs/`. Each run logs per-epoch loss and accuracy for training and testing phases:

```
[Run 1] Epoch [1/100] Train Loss: 1.2593, Acc: 63.42% | Val Loss: 0.9341, Acc: 78.01%
...
[Run 1] Best Val Acc: 81.53%
```

### 📂 Dataset Format

Expected directory structure:

```
/data_root/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```


### ✅ Features

* [x] Support for ViT and ResNet backbones
* [x] Easy integration of pretrained models (e.g., CLIP, MoCo)
* [x] Few-shot training and evaluation
* [x] Separate training modes: linear probing and full finetuning
* [x] Configurable optimizer, learning rate, and more


### 📜 Acknowledgements

This repository is inspired by and partially built upon:

* [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
* [MoCo: Momentum Contrast for Unsupervised Visual Representation Learning](https://github.com/facebookresearch/moco)
* [OpenAI CLIP](https://github.com/openai/CLIP)



