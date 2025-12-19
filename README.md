
# NeuroScan AI: Early Alzheimer’s Disease Detection

<div align="center">
    <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" alt="PyTorch">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</div>

## 🧠 Project Overview
NeuroScan AI is a deep learning system designed to detect early signs of Alzheimer's Disease (AD) and Mild Cognitive Impairment (MCI) from structural MRI scans. By analyzing 3D brain imagery, the model identifies subtle volumetric biomarkers that may indicate the onset of neurodegeneration.

This project implements the research methodologies described in [Recent Studies](https://www.nature.com/articles/s41598-022-20674-x) on utilizing CNNs for dementia classification.

## ✨ Features
- **3D Convolutional Neural Network**: Processes full MRI volumes for comprehensive analysis.
- **Web Interface**: User-friendly portal to upload and analyze scans instantly.
- **Multi-Format Support**: Compatible with medical NIfTI files (`.nii`, `.nii.gz`) and standard images (`.jpg`, `.png`).
- **Interactive Results**: Visual confidence scoring and detailed class explanations.

---

## 🚀 Quick Start Guide
This project is configured to run out-of-the-box with **Mock Data** for demonstration purposes, as the actual ADNI dataset is restricted.

### 1. Prerequisites
Ensure you have Python installed. Install dependencies:
```bash
pip install -r requirements.txt
pip install flask pillow nibabel
```

### 2. Launch the Web Application
Start the interface to use the model:
```bash
python web_app/app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

> **Note**: You can upload any `.jpg` or `.png` image to test the detection pipeline. The system auto-converts 2D images into 3D volumes for the model.

---

## 🛠️ Training the Model (Optional)
If you wish to retrain the model (e.g., after obtaining real data), follow these steps:

### A. Data Setup (Mock Mode)
Generate dummy NIfTI data to verify the training pipeline:
```bash
python create_mock_data.py
```

### B. Run Training
Execute the training loop using the local configuration:
```bash
python main.py --config config_local
```
*   **config_local.yaml**: Configured for quick CPU-based training on local machines.
*   **main.py**: The core training script.

---

## 📊 Dataset Information
This project is designed for the **ADNI (Alzheimer's Disease Neuroimaging Initiative)** dataset.
- **Real Data**: If you have access, download the ADNI dataset and preprocess it as described in `datasets/adni_3d.py`.
- **Mock Data**: For users without access, `create_mock_data.py` generates synthetic noise files to demonstrate that the code runs without errors.

> **⚠️ Important**: The "Mock Data" consists of random noise. Therefore, the model's predictions in this mode are for verifying the software functionality only and **do not represent real medical diagnosis**.

## 📁 Project Structure
```
├── web_app/               # Web Interface
│   ├── app.py             # Flask Backend
│   └── templates/         # Frontend HTML
├── saved_model/           # Trained Model Checkpoints
├── datasets/              # Data Loaders
├── models/                # Neural Network Architecture
├── main.py                # Training Entry Point
├── config_local.yaml      # Local Configuration
└── create_mock_data.py    # Mock Data Generator
```

## 📜 License
This project is for research and educational purposes.


## 📝 Acknowledgments
- **ADNI**: Alzheimer's Disease Neuroimaging Initiative. [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)
- **Flask**: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
- **NIfTI**: [https://nifti.nimh.nih.gov/](https://nifti.nimh.nih.gov/)

## 📝 Contact
For questions or feedback, please contact [Rajagopal](mailto:rajaganapthi1920@gmail.com).
# for runn 
python web_app/app.py



