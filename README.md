# Fusion-m6A: A Lightweight Hybrid Deep Learning Framework for m6A Site Prediction

<p align="center">
  <img width="4832" height="4864" alt="Hybrid-m6A" src="https://github.com/user-attachments/assets/7e272416-57a8-42e8-b7bf-fba449755c02" />
</p>


**Fusion-m6A** is a hybrid deep learning framework designed to predict **N⁶-methyladenosine (m⁶A)** RNA modification sites across multiple human tissues and cell lines. It integrates **Convolutional Neural Networks (CNNs)**, **Bidirectional Gated Recurrent Units (Bi-GRU)**, and an **attention mechanism** with an auxiliary **k-mer feature branch**, 
allowing simultaneous capture of **local motifs**, **long-range dependencies**, and **global compositional features**.  
This repository includes preprocessed benchmark datasets, pretrained models, and visualization tools for attention analysis and cross-tissue evaluation.


---

## 🚀 Key Features
- **Hybrid CNN-GRU-Attention architecture** for efficient and interpretable m⁶A prediction.  
- **Word2Vec-based embeddings** for nucleotide sequence representation.  
- **Cross-tissue generalization** analysis showing robust performance across 11 datasets.  
- **Integrated visualization tools** for:
  - Attention heatmaps and average attention profiles  
  - ROC and Precision–Recall curve analysis  
  - Computational efficiency and cross-test matrices  
- **Pretrained models** for each tissue and cell line for direct inference.

---

## 🧩 Reproducibility and Environment Setup
To ensure full reproducibility of results reported in the paper, we recommend setting up a dedicated conda environment with the exact package versions used during training and evaluation.
### Create a new environment
```bash
conda create -n fusion_m6a python=3.9
conda activate fusion_m6a
```
Install required libraries
All core dependencies are listed below (these were used for the published results):
| Library      | Version | Purpose                           |
| ------------ | ------- | --------------------------------- |
| Python       | 3.9     | Core environment                  |
| PyTorch      | 1.12.1  | Deep learning framework           |
| torchvision  | 0.13.1  | Model support utilities           |
| numpy        | 1.23.5  | Numerical computation             |
| pandas       | 1.5.3   | Data processing                   |
| scikit-learn | 1.2.2   | Evaluation metrics, preprocessing |
| gensim       | 4.3.1   | Word2Vec embeddings               |
| matplotlib   | 3.7.1   | Plotting and visualization        |
| seaborn      | 0.12.2  | Statistical visualization         |
| tqdm         | 4.65.0  | Training progress bar             |
| scipy        | 1.10.1  | Statistical computation           |
| PyYAML       | 6.0     | Configuration management          |
| h5py         | 3.8.0   | Model weight saving               |
| pickle5      | 0.0.12  | Object serialization              |

Install them directly:
```bash
pip install torch==1.12.1 torchvision==0.13.1 numpy==1.23.5 pandas==1.5.3 scikit-learn==1.2.2 gensim==4.3.1 matplotlib==3.7.1 seaborn==0.12.2 tqdm==4.65.0 scipy==1.10.1 PyYAML==6.0 h5py==3.8.0 pickle5==0.0.12
```
---

## **Comparison Model: MST-m6A (Transformer-based Model)**:
For this study, the official implementation of MST-m6A was **downloaded from the authors**’ **cbbl-skku-org** and all pretrained weights were loaded for inference:

**MST-m6A GitHub:** [https://github.com/cbbl-skku-org/MST-m6A](https://github.com/cbbl-skku-org/MST-m6A/)

**Citation for MST-m6A**
If you use MST-m6A for comparison, please cite the following paper:

> Su, Qiaosen, et al. *MST-m6A: a novel multi-scale transformer-based framework for accurate prediction of m6A modification sites across diverse cellular contexts.Journal of Molecular Biology 437.6 (2025): 168856.*
