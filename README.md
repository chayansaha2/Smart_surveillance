# 🛡️ Smart Surveillance: Deep Learning Threat Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, edge-deployable real-time surveillance and threat evaluation platform. The system leverages a lightweight multi-task spatio-temporal deep neural network to perform automated violence detection, dynamic Aggression Severity Indexing (ASI), temporal attention diagnostics (XAI), GDPR-compliant identity masking, and forensic incident reporting.

---

## ⚡ Key Highlights & Capabilities

* **Spatio-Temporal Feature Extraction:** Combines an inverted residual **MobileNetV2** backbone for high-speed spatial frame encoding (1,280-D representations) with a **2-Layer Bidirectional LSTM** (256 hidden units) to model sequence dynamics across 16-frame sliding windows.
* **Explainable AI (XAI) Attention Dynamics:** Employs a parameterized temporal attention head ($\alpha_t$) that weights the exact timestamps and frames responsible for anomalous classifications.
* **Multi-Task Aggression Severity Indexing (ASI):** Concurrently computes threat probabilities and categorizes incidents into structured severity tiers:
  * `Level 1: Low Agitation`
  * `Level 2: Physical Scuffle`
  * `Level 3: Critical Brawl`
* **GDPR-Compliant Identity Anonymization:** Integrated **YOLOv8** human tracking applies real-time Gaussian facial blurring to protect bystander privacy.
* **Automated Forensic Evidence Locker:** Auto-captures high-resolution frame snapshots during threshold breaches and compiles chronological incident logs into downloadable **PDF audit reports**.

---

## 📊 Benchmark & Performance

Evaluated on the 2,000-clip **Real-Life Violence Situations (RLVS)** benchmark:

| Metric | Proposed System | Baseline Comparison |
| :--- | :--- | :--- |
| **Validation Accuracy** | **97.75%** | CNN+LSTM Baselines (~88–91%) |
| **F1-Score** | **97.76%** | Vision Transformers (~96.25%) |
| **Model Size** | **~3.8M Parameters** | Lightweight & Edge-Deployable |
| **Inference Throughput** | **28–60 FPS** | Real-Time Optimization |

---

## 🔬 Mathematical Formulation

* **Spatial Feature Extraction:**
  $$f_t = \text{AdaptiveAvgPool2D}(\text{MobileNetV2}(x_t)), \quad f_t \in \mathbb{R}^{1280}$$

* **Bidirectional Temporal Context:**
  $$h_t = [\overrightarrow{\text{LSTM}}(f_t) \,\Vert{}\, \overleftarrow{\text{LSTM}}(f_t)], \quad h_t \in \mathbb{R}^{512}$$

* **Parameterized Temporal Attention:**
  $$\alpha_t = \frac{\exp(w^T \tanh(W_a h_t))}{\sum_{k=1}^T \exp(w^T \tanh(W_a h_k))}, \quad c = \sum_{t=1}^T \alpha_t h_t$$

* **Threat Probability & Multi-Task Head:**
  $$P_{\text{threat}} = \sigma(W_v c + b_v), \quad S_{\text{severity}} = \text{Softmax}(W_s c + b_s)$$

---

##  Quickstart & Local Deployment

