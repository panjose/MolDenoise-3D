# MolDenoise-3D: Streamlined 3D Molecular Representation Learning via Denoising
A PyTorch implementation of 3D molecular representation learning via denoising, with a streamlined, dependency-free architecture.

---

## 🚀 项目概述 (Overview)

本项目 **MolDenoise-3D** 是对原“pre-training-via-denoising”论文方法的 **精简（Streamlined）** 与 **重构（Refactoring）** 实现。

核心目标是实现一种高效、清晰的 **三维分子表示学习** 模型，通过 **去噪（Denoising）** 任务进行预训练，以捕获分子的三维几何和拓扑信息。

与原项目相比，本项目彻底移除了对大型训练框架（如 `pytorch-lightning`）的依赖，所有核心组件均独立实现，极大地提高了代码的**可读性、可维护性和灵活性**。

## ✨ 主要特点 (Features)

* **⚡️ 架构精简：** 彻底移除 `pytorch-lightning` 依赖，采用原生 PyTorch 实现训练流程。
* **📐 模块清晰：** 重构了庞大的 `Model` 类，将模型结构拆分，逻辑更加清晰易懂。
* **🛠️ 独立训练器：** 独立实现 `Trainer` 类，替代框架流程，提供简洁的训练/评估/日志管理。
* **🔄 新版适配：** 重新设计了 `Datasets` 类，完美适配最新 PyTorch 生态系统。
* **🧹 参数整理：** 对原项目中冗余的 `hparams` 参数进行了大幅删减与优化整理。

## ⚙️ 环境要求与安装 (Installation)

### 1. 克隆仓库

```bash
git clone https://github.com/panjose/MolDenoise-3D.git
cd MolDenoise-3D
````

### 2\. 创建并激活环境

建议使用 Conda 创建隔离环境：

```bash
conda create -n moldenoise python=3.12
conda activate moldenoise
```

### 3\. 安装依赖

本项目依赖主要包括 PyTorch、RDKit（用于分子处理）及其他基础科学计算库。

```bash
pip install -r requirements.txt
# 或者手动安装核心依赖：
# pip install torch rdkit numpy pandas
```

## 📚 使用方法 (Usage)

### 1\. 数据准备 (Data Preparation)

将分子数据（如 SDF/SMILES 文件）放置于 `./data` 目录下，或直接运行后续脚本。

`pcq`数据集 http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2_xyz.zip

### 2\. 模型训练 (Training)

通过运行主训练脚本启动预训练流程，或从`./examples/pretrain/models_3d`目录中寻找`frad.sh`直接运行。

```bash
python -B pretrain_repr.py examples/pretrain/models_3d/frad.yaml
```

  * 参数配置位于 `./examples/pretrain/models_3d/` 目录下。

### 3\. 模型评估与下游微调 (Evaluation)

主要通过微调qm9数据集，通过运行主训练脚本启动预训练流程，或从`./examples/finetune/models_3d`目录中寻找`frad.sh`直接运行。

```bash
python -B finetune_repr.py examples/finetune/models_3d/frad.yaml
```

  * 参数配置位于 `./examples/finetune/models_3d/` 目录下。

## 📂 项目结构 (Project Structure)

```
MolDenoise-3D/
├── examples/             # 训练和模型配置参数 (YAML) 以及运行指令 (SH)
├── data/                 # 存放数据文件
│   ├── raw/
│   └── processed/
├── save                  # 训练保存的参数和日志
├── src/                  # 核心源代码
│   ├── models/           # 模型定义（包含精简后的模块）
│   ├── datasets/         # 重新实现的 Dataset 类
│   ├── hparams/          # 参数读取处理 
│   ├── trainer/          # 独立实现的 Trainer 类
|   ├── __init__.py
|   └── utils.py          # 辅助工具
├── pretrain_repr.py      # 预训练入口文件
├── finetune_repr.py      # 微调入口文件
├── requirements.txt      # Python 依赖列表
└── README.md             # 项目说明文件 (本文档)
```

## 🤝 贡献与致谢 (Contribution & Acknowledgement)

欢迎提交 Issue 和 Pull Request 来改进本项目。

本项目基于 shehzaidi/pre-training-via-denoising 的核心思想进行重构，在此对原作者表示感谢。
