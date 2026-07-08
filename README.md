# RFM-UNet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

---

## 📖 Introduction

1.  This is the code for our paper: RFM-UNet: Hybrid Frequency-Mamba UNet for Remote Sensing Road Extraction

---

## 💻 System Requirements

This model is optimized for **accuracy-critical applications** in geosciences. Due to the sophisticated attention mechanisms and high-resolution inputs, it requires a robust hardware environment.

*   **OS:** Linux (Ubuntu 20.04/22.04 recommended).
*   **GPU:** NVIDIA GPU with **Compute Capability ≥ 8.0** (Ampere architecture or newer).
*   **VRAM:** 
    *   **Training:** ≥ 24GB (e.g., RTX 3090/4090/5090) is strongly recommended.
    *   **Inference:** ≥ 12GB.
*   **CUDA:** 12.4 (Strictly required for the provided installation steps).

---

## 🛠️ Installation

To ensure reproducibility, please follow these steps strictly to configure the Mamba environment.

### 1. Clone the repository
```bash
git clone https://github.com/FF7CA/RFMUNet.git
cd RFMUNet
```

### 2. Create Environment
```bash
conda create -n aerith python=3.12
conda activate aerith
```

### 3. Install PyTorch (CUDA 12.4)

Note: We use PyTorch 2.6.0 which is compatible with Mamba 2.2.4.
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install Mamba-SSM & Dependencies
This step requires nvcc (CUDA compiler) to be available in your path.
```bash
# Install core libraries
pip install -r requirements.txt

# Install Mamba components
pip install causal-conv1d==1.6.0
pip install mamba-ssm==2.2.4
```

### 5. Install VMamba
This step requires nvcc (CUDA compiler) to be available in your path.
```bash
# Install VMamba libraries
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```

## :rocket:Training

```
python train_supervision.py -c ./config/Mass/RFMUNet.py
```

```
python train_supervision.py -c ./config/DPGB/RFMUNet.py
```

```
python train_supervision.py -c ./config/CHN6/RFMUNet.py
```

## :100: Testing

```
python road_seg_test.py -c ./config/Mass/RFMUNet.py -o /root/results/Mass/RFMUNet --rgb -t 'lr'
```

```
python road_seg_test.py -c ./config/DPGB/RFMUNet.py -o /root/results/DPGB/RFMUNet --rgb -t 'lr'
```

```
python road_seg_test.py -c ./config/CHN6/RFMUNet.py -o /root/results/CHN6/RFMUNet --rgb -t 'lr'
```

---

## 📂 Data Preparation

### Download the dataset

```
The datasets used and analyzed during the current study are available from the following sources: Massachusetts road dataset: https://www.cs.toronto.edu/~vmnih/data/; DeepGlobe road dataset: http://deepglobe.org/challenge.html; CHN6-CUG road dataset: https://grzy.cug.edu.cn/zhuqiqi/zh_CN/index.html.
```

### Organize the data as follows:

```text
datasets/
├── Massachusetts road/
│   ├── test/
│   ├── train/
│   ├── val/
└── DeepGlobe road dataset/
│   ├── test/
│   ├── train/
│   ├── val/
└── CHN6-CUG/
│   ├── train/
│   ├── val/
```
## 🤝 Acknowledgement

Our training scripts comes from [GeoSeg](https://github.com/WangLibo1995/GeoSeg). Thanks for the author's open-sourcing code.

- [GeoSeg(UNetFormer)](https://github.com/WangLibo1995/GeoSeg)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

