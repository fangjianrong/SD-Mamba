# SD-Mamba: Spatial-Dynamic Mamba for Hyperspectral Image Classification

本项目包含了 **SD-Mamba** 模型的完整 PyTorch 实现与训练/测试脚本。

## ⚙️ 1. 环境准备 (Environment Setup)

在开始之前，请确保你的环境中安装了以下依赖库：

* Python >= 3.8
* PyTorch >= 1.10.0 (支持 CUDA)
* NumPy, SciPy, Pandas
* scikit-learn
* h5py (用于兼容不同版本的 `.mat` 数据集)

你可以通过以下命令快速安装基础依赖：
```bash
pip install torch numpy scipy pandas scikit-learn h5py

├── data/
│   ├── Indian_pines_corrected.mat   # IP 数据
│   ├── Indian_pines_gt.mat          # IP 标签
│   ├── PaviaU.mat                   # PU 数据
│   ├── PaviaU_gt.mat                # PU 标签
│   ├── Houston.mat                  # HU13 数据
│   ├── Houston_gt.mat               # HU13 标签
│   ├── WHU_Hi_HongHu.mat            # HH 数据
│   ├── WHU_Hi_HongHu_gt.mat         # HH 标签
│   ├── WHU_Hi_HanChuan.mat          # HC 数据
│   └── WHU_Hi_HanChuan_gt.mat       # HC 标签
├── SD_Mamba_model.py
├── data_utils.py
└── train.py
训练脚本已经高度封装，直接运行 train.py 即可自动完成数据划分、模型训练以及模型验证的全流程。
python train.py
