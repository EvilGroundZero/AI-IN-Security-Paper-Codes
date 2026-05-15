# AI-Based Network Intrusion Detection — NSL-KDD Experiment

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: NSL-KDD](https://img.shields.io/badge/Dataset-NSL--KDD-purple)](https://www.unb.ca/cic/datasets/nsl.html)

> **Course:** Applications of Artificial Intelligence in Cyber Security  
> **Author:** Abdulrahman Almaslamani — Student ID: 202517297  
> **Paper:** *Artificial Intelligence and Generative Models for Network Intrusion Detection: A Mixed-Methods Comparative Study on the NSL-KDD Dataset*

---

## Overview

This repository contains the complete experimental code, generated figures, and results for a research paper comparing three machine learning models for AI-driven network intrusion detection on the NSL-KDD benchmark dataset:

| Model | Binary Accuracy | Binary F1 | Multi-Class F1 (Wt) | Train Time |
|---|---|---|---|---|
| **Random Forest** | 78.09% | 76.51% | 71.57% | **2.8s** |
| **SVM** | 79.62% | 78.34% | **72.66%** | 340.7s |
| **MLP (Neural Net)** | **80.69%** | **79.78%** | 71.37% | 37.2s |

> **Key finding:** All models achieve high precision (96–98%) but moderate recall (63–68%) on KDDTest+. This reflects the deliberate generalisation challenge of the official test set — not a model failure — and directly motivates generative AI augmentation for minority classes (R2L, U2R).

---

## Repository Structure

```
AI-IN-Security-Paper-Codes/
│
├── experiment.py               # ← Main script: trains all models, runs all evaluation, saves all figures
│
├── charts/
│   ├── fig1_class_distribution.png          # Class distribution: training vs test set
│   ├── fig2_confusion_matrix_rf_binary.png  # RF confusion matrix (binary)
│   ├── fig3_performance_comparison.png      # All metrics side-by-side comparison
│   ├── fig4_roc_curves.png                  # ROC curves for all 3 models
│   ├── fig5_feature_importance.png          # Top 15 RF feature importances
│   ├── fig6_training_time.png               # Training time comparison
│   ├── fig7_multiclass_f1.png               # Per-class F1 (multi-class)
│   └── fig8_confusion_matrix_rf_multiclass.png  # RF confusion matrix (5-class)
│
├── results/
│   └── results_summary.csv     # Numeric results for all models, both tasks
│
├── README.md
└── LICENSE
```

---

## Getting Started

### 1. Download the Dataset

The NSL-KDD dataset is **not included** in this repository. Download it for free from one of these sources:

**Option A — Official (recommended):**
```
https://www.unb.ca/cic/datasets/nsl.html
```
Download `KDDTrain+.txt` and `KDDTest+.txt`.

**Option B — Direct wget (Linux/macOS/Kali):**
```bash
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt
```

**Option C — Kaggle:**
```bash
kaggle datasets download -d hassan06/nslkdd
```

Place both `.txt` files in the **same folder** as `experiment.py`.

---

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

On Kali Linux or systems with PEP 668 protection:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn --break-system-packages
```

Python 3.10 or higher is required.

---

### 3. Run the Experiment

```bash
python3 experiment.py
```

**What happens:**
1. Loads and preprocesses KDDTrain+.txt and KDDTest+.txt
2. Trains Random Forest, SVM, and MLP on binary classification (Normal vs Attack)
3. Trains the same three models on multi-class classification (5 categories)
4. Prints a full results table to the terminal
5. Saves 8 PNG figures and `results_summary.csv` to the current directory

> ⏱ **Expected runtime:** RF (~3s), MLP (~40s), SVM (~6 min). SVM is slow due to quadratic scaling on 125,973 training samples. Total: ~7–8 minutes.

---

## Results Summary

### Binary Classification (Normal vs Attack)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Random Forest | 78.09% | 96.66% | 63.31% | 76.51% | 0.9556 |
| SVM | 79.62% | 97.71% | 65.37% | 78.34% | 0.9472 |
| MLP | **80.69%** | 97.36% | **67.58%** | **79.78%** | 0.9332 |

### Multi-Class Classification (5 Categories, Weighted Average)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 76.01% | 82.31% | 76.01% | 71.57% |
| SVM | **77.74%** | 80.90% | **77.74%** | **72.66%** |
| MLP | 75.84% | 80.90% | 75.84% | 71.37% |

### Per-Class F1 — Random Forest (Multi-Class)

| Class | Precision | Recall | F1 | Test Samples |
|---|---|---|---|---|
| Normal | 0.660 | 0.974 | 0.787 | 9,711 |
| DoS | 0.962 | 0.801 | 0.874 | 7,167 |
| Probe | 0.871 | 0.665 | 0.754 | 2,421 |
| R2L | 0.990 | 0.034 | 0.065 | 2,885 |
| U2R | 0.667 | 0.030 | 0.057 | 67 |

---

## Why Is Recall ~63%?

This is the expected result when evaluating on the **official KDDTest+ set**. KDDTest+ is deliberately harder than KDDTrain+:

- **R2L** has only 995 training samples but **2,885 test samples** (3x more)
- Many R2L subcategories in KDDTest+ were **never seen during training**
- Studies reporting 99% accuracy typically evaluate on a random split of KDDTrain+ — avoiding this challenge entirely

The high precision (96–98%) shows models rarely generate false alarms. The recall gap identifies exactly where generative AI augmentation (GANs, VAEs) would have the greatest impact.

---

## Models and Hyperparameters

```python
# Random Forest
RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

# SVM
SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)

# MLP
MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
```

All random seeds fixed at 42 for full reproducibility.

---

## Attack Category Mapping

| Category | Attack Types in Dataset |
|---|---|
| **Normal** | normal |
| **DoS** | neptune, back, land, pod, smurf, teardrop, apache2, udpstorm, processtable, worm |
| **Probe** | satan, ipsweep, nmap, portsweep, mscan, saint |
| **R2L** | ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster, sendmail, named, snmpgetattack, snmpguess, xlock, xsnoop, httptunnel |
| **U2R** | buffer_overflow, loadmodule, perl, rootkit, ps, sqlattack, xterm |

---

## Citation

If you use this code in your work, please cite:

```bibtex
@misc{almaslamani2025ids,
  author    = {Abdulrahman Almaslamani},
  title     = {Artificial Intelligence and Generative Models for Network Intrusion Detection:
               A Mixed-Methods Comparative Study on the NSL-KDD Dataset},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/EvilGroundZero/AI-IN-Security-Paper-Codes}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **NSL-KDD Dataset:** Canadian Institute for Cybersecurity, University of New Brunswick — https://www.unb.ca/cic/datasets/nsl.html
- **scikit-learn:** Pedregosa et al., JMLR 12, pp. 2825–2830, 2011
