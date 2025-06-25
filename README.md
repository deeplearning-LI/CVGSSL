# CVGSSL

## The paper "CLIP-Vision Guided Few-Shot Metal Surface Defect Recognition" has been published in IEEE Transactions on Industrial Informatics.

## 🧠 CVGSSL-Finetune & Linear Evaluation

本项目基于 PyTorch 实现了一个用于图像识别的微调（Finetune）与线性评估（Linear Evaluation）训练框架，支持 ResNet 和 Vision Transformer (ViT) 架构，可用于少样本（Few-shot）图像识别任务，结合多模态预训练模型（如 CLIP）进行特征蒸馏。

---

### 📁 项目结构

```
.
├── finetune_and_lincls.py     # 主训练脚本（Finetune & Linear）
├── builder/
│   └── clip_base.py           # 模型构建模块（自定义 CLIP 蒸馏结构）
├── data_aug/
│   ├── loader.py              # 自定义 GaussianBlur 和 Dataset 加载器
│   └── transform.py           # 数据增强策略
├── FSDataset.py               # Few-shot 数据集构建类
├── pretrained/                # 预训练权重目录
├── logs/                      # 训练日志保存路径
└── README.md
```

---

### 🚀 快速开始

#### 1. 安装依赖

```bash
pip install torch torchvision timm numpy
```

#### 2. 运行训练脚本

```bash
python finetune_and_lincls.py \
  --data /path/to/dataset \
  --pretrained /path/to/pretrained_model.pth.tar \
  --arch vit_base_patch16_224 \
  --builder cgssl \
  --softmax_lincls False \
  --finetune_model True \
  --optimizer sgd \
  --lr 0.003 \
  --shot 4 \
  --n 5 \
  --class_num 6 \
  --epochs 100 \
  --gpu 0
```

---

### 🔧 参数说明

| 参数                 | 说明                               | 示例                           |
| ------------------ | -------------------------------- | ---------------------------- |
| `--data`           | 数据集根目录，包含 `train/` 和 `test/` 文件夹 | `./datasets/NEU`             |
| `--pretrained`     | 预训练模型权重路径（支持MoCo/CLIP蒸馏）         | `./pretrained/model.pth.tar` |
| `--arch`           | 模型架构，支持 ResNet、ViT               | `vit_base_patch16_224`       |
| `--softmax_lincls` | 是否为线性评估模式（冻结除最后一层外参数）            | `True / False`               |
| `--finetune_model` | 是否进行全模型微调                        | `True`                       |
| `--optimizer`      | 优化器类型                            | `sgd` 或 `adam`               |
| `--lr`             | 学习率                              | `0.003`                      |
| `--shot`           | 每类训练样本数（Few-shot）                | `4`                          |
| `--n`              | 重复训练次数（用于平均结果）                   | `5`                          |
| `--class_num`      | 类别数                              | `6`                          |
| `--epochs`         | 总训练轮数                            | `100`                        |
| `--gpu`            | 使用的 GPU ID                       | `0`                          |

---

### 📊 输出日志示例

训练日志将自动保存在 `logs/` 下，包含每轮的训练与验证损失、准确率：

```
[Run 1] Epoch [1/100] Train Loss: 1.2593, Acc: 63.42% | Val Loss: 0.9341, Acc: 78.01%
[Run 1] Epoch [2/100] ...
...
[Run 1] Best Val Acc: 81.53%
```

---

### 🧪 模型结构与蒸馏说明（示意）

本框架支持将 ViT Backbone 与 CLIP 图像特征结合，进行蒸馏训练，或作为初始化参数来源。结构如下：

```
Backbone (ResNet / ViT)
        ↓
  Feature Embedding
        ↓
 Projection Head (MLP or Linear)
        ↓
     Softmax 分类器
```

---

### 📂 数据格式说明

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

---

### ✨ TODO

* [ ] 支持混合精度训练（FP16）
* [ ] 支持 EMA 模型平均
* [ ] 集成 TensorBoard/W\&B 可视化
* [ ] 迁移至分布式训练结构

---

### 📮 引用与致谢

本项目部分借鉴以下库：

* [timm](https://github.com/huggingface/pytorch-image-models)
* [MoCo](https://github.com/facebookresearch/moco)
* [CLIP](https://github.com/openai/CLIP)


