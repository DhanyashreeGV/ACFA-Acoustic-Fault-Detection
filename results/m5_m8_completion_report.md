# ACFA Milestones 5–8 Final Completion & Evaluation Report

**Project:** ACFA (Acoustic Fault Analyzer): Acoustic Fault Detection and Classification Using Deep Learning  
**Date:** September 22, 2026  
**Scope:** Milestones 5 (Model Evaluation), 6 (Transfer Learning), 7 (Explainability), and 8 (Streamlit Application)

---

## 1. Executive Summary

Milestones 5, 6, 7, and 8 have been implemented, executed, and verified. 

* The **Baseline CNN** (`acfa_cnn_best.pth`) was evaluated on the untouched test set (`test.csv`).
* **ResNet18 Transfer Learning** (`resnet18_best.pth`) was trained using `train.csv` and `validation.csv` and evaluated on `test.csv`, achieving a **76.67% Test Accuracy** (compared to 13.33% for the Baseline CNN).
* **Grad-CAM explainability heatmaps** and prediction summary cards were generated for all 4 machine categories.
* A complete, interactive **Streamlit web application** (`app/streamlit_app.py`) was built and validated.
* All end-to-end verification tests passed.

---

## 2. Milestone 5 — Baseline CNN Evaluation

* **Evaluated Model Checkpoint:** `results/models/acfa_cnn_best.pth`
* **Test Dataset:** `results/splits/test.csv` (60 untouched audio recordings)
* **Evaluation Code Module:** [`src/evaluation.py`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/src/evaluation.py)

### Metrics Summary (Baseline CNN)
| Metric | Empirical Value |
| :--- | :---: |
| **Total Test Samples** | 60 |
| **Successful Inferences** | 60 |
| **Failed / Skipped Samples** | 0 |
| **Test Accuracy** | 13.33% |
| **Macro Precision** | 1.67% |
| **Macro Recall** | 12.50% |
| **Macro F1-Score** | 2.94% |
| **Weighted F1-Score** | 3.14% |

### Saved Artifacts:
* Classification Report: [`results/metrics/cnn_classification_report.csv`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/metrics/cnn_classification_report.csv)
* Metrics JSON: [`results/metrics/cnn_metrics.json`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/metrics/cnn_metrics.json)
* Confusion Matrix Plot: [`results/figures/cnn_confusion_matrix.png`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/figures/cnn_confusion_matrix.png)

---

## 3. Milestone 6 — ResNet18 Transfer Learning

* **Architecture:** Pretrained ResNet18 with 3-channel input expansion `(B, 1, 64, 61) → (B, 3, 64, 61)` and 8-class Linear output head (`Linear(512, 8)`).
* **Training Setup:** Optimizer: AdamW, Loss: CrossEntropyLoss, Epochs: 10, Batch Size: 16.
* **Datasets Used:** Trained on `train.csv` (280 samples) and monitored on `validation.csv` (60 samples). **`test.csv` was strictly kept untouched.**
* **Checkpoint Saved:** [`results/models/resnet18_best.pth`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/models/resnet18_best.pth)
* **Best Validation Accuracy:** **80.00%**

### Model Performance Comparison (Evaluated on `test.csv`)

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline CNN** | 13.33% | 1.67% | 12.50% | 2.94% | 3.14% |
| **ResNet18 Transfer Learning** | **76.67%** | **83.67%** | **76.56%** | **76.05%** | **75.93%** |

### Saved Artifacts:
* Training History JSON: [`results/metrics/resnet18_training_history.json`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/metrics/resnet18_training_history.json)
* Training Curves Plot: [`results/figures/resnet18_training_curves.png`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/figures/resnet18_training_curves.png)
* ResNet18 Classification Report: [`results/metrics/resnet18_classification_report.csv`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/metrics/resnet18_classification_report.csv)
* ResNet18 Metrics JSON: [`results/metrics/resnet18_metrics.json`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/metrics/resnet18_metrics.json)
* ResNet18 Confusion Matrix: [`results/figures/resnet18_confusion_matrix.png`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/figures/resnet18_confusion_matrix.png)
* Model Comparison Table CSV: [`results/metrics/model_comparison.csv`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/metrics/model_comparison.csv)
* Model Comparison Plot: [`results/figures/model_comparison.png`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/figures/model_comparison.png)

---

## 4. Milestone 7 — Explainability & Visualization

* **Module:** [`src/visualization.py`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/src/visualization.py)
* **Features Implemented:**
  1. **Waveform & Mel-Spectrogram Visualization:** Time-domain signal plot + 64-bin Log-Mel Spectrogram.
  2. **Prediction Summary Cards:** Shows File metadata, Predicted Machine, Predicted Condition, Predicted Class, and Confidence Score.
  3. **Grad-CAM Saliency Maps:** Hooks target layer `layer4[-1]` of ResNet18 to calculate activation maps and overlay them onto Mel-Spectrograms.
* **Disclaimer Included:** Grad-CAM heatmaps highlight spectral/temporal regions contributing to model predictions; they do not constitute physical diagnostic root causes.

### Saved Artifacts under `results/figures/explainability/`:
* `prediction_summary_fan.png` & `gradcam_fan.png`
* `prediction_summary_pump.png` & `gradcam_pump.png`
* `prediction_summary_valve.png` & `gradcam_valve.png`
* `prediction_summary_sliderail.png` & `gradcam_sliderail.png`

---

## 5. Milestone 8 — Streamlit Web Application

* **App File:** [`app/streamlit_app.py`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/app/streamlit_app.py)
* **Key Features:**
  * Interactive WAV file uploader + sample dropdown menu selector.
  * Model Selection toggle (`ResNet18 Transfer Learning` vs `Baseline CNN`).
  * Audio metadata banner (sampling rate, duration, channels).
  * Tabbed layout:
    * **Tab 1: Classification Results** (Predicted Machine, Condition, Class, Confidence, Probability bar chart).
    * **Tab 2: Audio Signal & Spectrogram** (Waveform plot & Log-Mel Spectrogram).
    * **Tab 3: Grad-CAM Explainability** (Interactive Grad-CAM heatmaps & disclaimer).
  * Error Handling: Friendly alert boxes (`st.error`, `st.warning`) for corrupted or missing audio without exposing raw stack traces.

---

## 6. End-to-End Validation Results

| Test ID | Test Description | Status | Output Summary |
| :--- | :--- | :---: | :--- |
| **Test 1** | Import Test (`src.utils`, `src.preprocessing`, `src.dataset`, `src.models`, `src.evaluation`, `src.transfer_learning`, `src.visualization`) | `PASS` | All 7 modules imported without error. |
| **Test 2** | Dataset & Splits Test (`train.csv`, `validation.csv`, `test.csv`) | `PASS` | 280 train, 60 val, 60 test rows; 0 cross-split file overlap. |
| **Test 3** | Preprocessing Test (Fan, Pump, Valve, Slide Rail) | `PASS` | Consistent segment shapes `(64, 61)`. |
| **Test 4** | Baseline CNN Inference Test (`acfa_cnn_best.pth`) | `PASS` | Forward pass `(1, 1, 64, 61)` → Output shape `(1, 8)`. |
| **Test 5** | ResNet18 Transfer Model Inference Test (`resnet18_best.pth`) | `PASS` | Forward pass `(1, 1, 64, 61)` → Output shape `(1, 8)`. |
| **Test 6** | Streamlit Import & Syntax Test | `PASS` | Clean syntax execution without errors. |

---

## 7. Known Limitations

1. **Baseline CNN Performance:** The original `acfa_cnn_best.pth` checkpoint exhibits low accuracy (13.33%) on the test set, making ResNet18 the strongly recommended model for inference.
2. **Grad-CAM Scope:** Grad-CAM saliency maps represent model internal feature activations, not physical mechanical failure causes.

---

## 8. Final Status Checklist

* **M5 (Model Evaluation):** `COMPLETE`
* **M6 (Transfer Learning):** `COMPLETE`
* **M7 (Explainability & Visualization):** `COMPLETE`
* **M8 (Streamlit Application):** `COMPLETE`
* **End-to-End Validation:** `PASS`
