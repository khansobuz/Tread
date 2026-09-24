# Temporal Relational Anomaly Modeling for Weakly Supervised Video Anomaly Detection

This repository contains the implementation of **Temporal Relational Anomaly Modeling (TRAM)** for weakly supervised video anomaly detection. The framework uses a frozen Multimodal Large Language Model (MLLM) to capture temporal relational anomaly knowledge from neighboring video clips and transfers this knowledge to a lightweight student network. The proposed approach focuses on modeling how anomalous events evolve within their surrounding temporal context rather than relying only on isolated anomaly scores. During inference, the MLLM teacher is removed, and only the lightweight student model is used for efficient anomaly detection.


## Framework Overview
<img width="3831" height="1125" alt="Main_fig" src="https://github.com/user-attachments/assets/826ab7c6-73c5-4507-a377-ee55cc30b358" />


## Installation

Install the required dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

Alternatively, the main dependencies can be installed manually:

```bash
pip install torch torchvision transformers qwen-vl-utils opencv-python pillow pandas scikit-learn scipy
pip install git+https://github.com/openai/CLIP.git
```

## Repository Files

```text
MLLM_features_UCF.py       # MLLM feature extraction for UCF-Crime
MLLM_features_XD.py        # MLLM feature extraction for XD-Violence
main_UCF.py               # Training and evaluation on UCF-Crime
main_XD.py                # Training and evaluation on XD-Violence
efficiency_ucf.py         # Efficiency evaluation
Qualtative.py             # Qualitative visualization
requirements.txt          # Required Python packages
checkpoint/               # Model checkpoints and figures
```

## Datasets

The framework supports the following datasets:

[UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/)

[XD-Violence](https://roc-ng.github.io/XD-Violence/)

Please prepare the corresponding video data, pre-extracted features, training/testing lists, and ground-truth annotations before training.


## MLLM Knowledge Extraction

The frozen MLLM is used to extract temporal relational anomaly knowledge from raw videos.

### UCF-Crime

```bash
python LLM_features_UCF.py
```

### XD-Violence

```bash
python LLM_features_XD.py
```

## Training and Evaluation

### UCF-Crime

```bash
python main_UCF.py
```

### XD-Violence

```bash
python main_XD.py
```

## Efficiency Evaluation

```bash
python efficiency_ucf.py
```

This script evaluates computational efficiency, including model parameters, GPU memory consumption, and processing time.

## Inference

During training, the lightweight student learns temporal relational anomaly knowledge from the frozen MLLM teacher. During inference, only the student model is used, enabling efficient deployment without the computational cost of the MLLM.

## Requirements

* Python 3.10 or later
* PyTorch
* CUDA-compatible GPU recommended
* Transformers
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* SciPy
* Pillow

## Citation

If you find this repository useful in your research, please consider citing the corresponding paper.
