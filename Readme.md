# 🏭 Industrial Quality Control: Computer Vision & Anomaly Detection

An end-to-end computer vision pipeline designed for automated defect detection on manufacturing surfaces. This project moves beyond standard classification by benchmarking classical machine learning against deep learning and introducing an unsupervised anomaly detection engine for zero-shot defect identification.

## 🚀 Business Case & Impact
In manufacturing, manual visual inspection is slow and error-prone. This project automates quality assurance by identifying surface defects (crazing, inclusions, patches, pitted surfaces, rolled-in scale, scratches) in real-time. 

Crucially, this project includes a **Trade-off Analysis (Accuracy vs. Inference Speed)** to determine the most viable model for deployment on hardware-constrained factory edge devices.

### Key Technical Achievements:
* **Transfer Learning**: Fine-tuned **MobileNetV2** for feature extraction, achieving **93.89% accuracy** on multi-class defect categorization.
* **Unsupervised Anomaly Detection**: Engineered a **Convolutional Autoencoder** to identify novel defects (Zero-Shot Detection) using Mean Squared Error (MSE) reconstruction thresholds.
* **Algorithm Benchmarking**: Conducted rigorous comparative analysis between Deep Learning (CNNs) and Classical ML (Support Vector Machines, Random Forests) utilizing flattened pixel vectors.

## 📊 Model Performance & Benchmarks

| Model | Task Type | Accuracy | Inference Time (ms) | Key Strength |
| :--- | :--- | :--- | :--- | :--- |
| **Convolutional Autoencoder** | Unsupervised Anomaly Detection | **96.77%** | 424.22 ms | Zero-Shot Detection |
| **MobileNetV2 (CNN)** | Supervised Multi-Class | **93.89%** | 2108.24 ms | High Spatial Awareness |
| **Random Forest** | Supervised Multi-Class | 73.33% | **6.96 ms** | Ultra-Low Latency |
| **Support Vector Machine** | Supervised Multi-Class | 71.11% | 24.70 ms | Baseline Simplicity |

*Note: MobileNetV2 provides the highest classification accuracy, while Random Forest offers the lowest latency for strict real-time constraints.*

## 📂 Repository Structure
```text
├── dl-project-ind.ipynb   # Main Kaggle notebook containing all pipelines
├── README.md              # Project documentation
