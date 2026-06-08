# 🩸 HemoVision AI - Clinical Anemia Screening

> ⚠️ **Disclaimer:** HemoVision is currently in a **very early Proof-of-Concept (PoC) phase**. This software is strictly for research and educational purposes and has not been validated for actual clinical diagnosis or medical decision-making.

## 📖 About This Project
This repository was built as a hands-on learning exercise to deeply explore the practical applications of **Machine Learning**. The goal was to go beyond simple tutorials and build a complete, end-to-end deep learning pipeline. It represents an engineering journey covering everything from curating datasets and training Convolutional Neural Networks (CNNs) to deploying the final model inside a custom-built desktop application. 

---

HemoVision is a lightweight, edge-optimized deep learning desktop application designed for non-invasive rapid clinical triage. It predicts systemic hemoglobin (Hgb) levels by analyzing the conjunctival pallor of the lower eyelid (palpebral tissue) from standard mobile phone images.

Built to identify severe systemic anemia in low-resource environments, the application mathematically maps tissue redness to clinical hemoglobin values without requiring a physical blood draw.

## ✨ Key Features
* **Continuous Regression Architecture:** Instead of binary classification, the AI outputs an exact continuous predicted value (e.g., 11.2 g/dL) trained against ground-truth clinical data.
* **Edge-Optimized Engine:** Powered by a **MobileNetV2** backend utilizing Transfer Learning, allowing the entire neural network to run locally on a standard desktop CPU/GPU without internet access.
* **Interactive Clinical Cropping:** A custom PyQt6 interface featuring a precision "Rubber Band" selection tool. This allows clinicians to manually isolate pure palpebral tissue, mathematically shielding the AI from facial, skin, and eyelash noise.
* **Intelligent Image Preprocessing:** Automatically handles EXIF rotation data from mobile cameras and utilizes strict string-matching filters to guarantee dataset purity.

## 🔬 Clinical Performance
The underlying AI model was rigorously tested using **5-Fold Cross-Validation** on a heavily curated dataset of 112 patients.
* **Best Validation Fold:** 1.86 g/dL Mean Absolute Error (MAE)
* **Overall Average:** ~2.0 g/dL MAE

>***Note**: HemoVision is a rapid screening/triage tool, not a diagnostic replacement for a Complete Blood Count (CBC) test. It is designed to flag severe systemic anemia (Hgb < 8.0 g/dL) for immediate medical escalation.*

## 🛠️ Technology Stack
* **Deep Learning:** TensorFlow / Keras (MobileNetV2)
* **Computer Vision:** OpenCV, NumPy
* **Desktop Interface:** PyQt6
* **Language:** Python 3.x

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/denreiangeles-code/hemovision-ai.git]

   cd hemovision-ai/hemovision_desktop_app
   ```
2. **Install the required dependencies:**
    - It is highly recommended to use a virtual environment.
    ```bash
    pip install PyQt6 opencv-python numpy tensorflow
    ```
3. **Ensure the AI Engine is present:**
    - Verify that `hemovision_final.keras` is located in the exact same directory as `main.py`.
4. **Launch the application:**

    ```bash
    python main.py
    ```
## 📸 Clinical Capture Guidelines
For the AI to accurately calculate hemoglobin, the uploaded images must adhere to strict clinical capture rules:
1. Pull Down: Gently pull the lower eyelid down completely to expose the inner pink palpebral tissue.

2. Lighting: Use the smartphone flash or face bright natural daylight. Avoid warm overhead bulbs.

3. Distance: Hold the camera 4-6 inches away from the eye.

4. No Filters: Ensure native camera skin smoothing and beauty filters are completely disabled.

>***Note**: For test purposes, you can use the samples in `training_model/India` and `training_model/Italy`.*

## 🗺️ Product Roadmap & Future Improvements
While Version 1.0 successfully proves the core clinical concept, the following architectural upgrades are planned for subsequent releases:

- **Automated ROI Extraction (Auto-Cropping)**: Transitioning from the manual "Rubber Band" clinical crop to an automated facial landmark detection pipeline (e.g., MediaPipe Face Mesh). This will instantly locate and isolate the palpebral conjunctiva, streamlining the user workflow while mathematically safeguarding the AI from facial noise.

- **Dataset Expansion & Diversification**: Scaling the foundational 112-patient dataset to thousands of curated clinical samples. Expanding the training data across diverse demographic profiles, ambient lighting conditions, and varying smartphone camera hardware will drastically improve the model's real-world generalization.

- **Heavyweight Architecture Scaling**: As the dataset grows large enough to safely prevent catastrophic overfitting, the backend visual engine will be upgraded from MobileNetV2 to deeper, highly optimized architectures (such as EfficientNetB0 or ResNet50) to capture even more complex visual biomarkers.