# ACFA (Acoustic Fault Analyzer): Acoustic Fault Detection and Classification Using Deep Learning

ACFA is an industrial acoustic fault detection system designed to analyze factory machine sound recordings and identify anomalous operating conditions using deep learning CNN models and transfer learning.

## Project Structure

```
ACFA-DL/
├── app/
│   └── streamlit_app.py     # Streamlit Web Application (M8)
├── src/                     # Core Python Package
│   ├── __init__.py
│   ├── utils.py             # Utility functions and label mappings
│   ├── preprocessing.py     # Audio loading, normalization, segmentation, Log-Mel Spectrogram
│   ├── dataset.py           # PyTorch Dataset & DataLoader
│   ├── models.py            # Baseline CNN Model architecture & loader
│   ├── evaluation.py        # Evaluation pipeline & metric computation (M5)
│   ├── transfer_learning.py # ResNet18 Transfer Learning training & comparison (M6)
│   └── visualization.py     # Prediction summary cards & Grad-CAM explainability (M7)
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_model_evaluation.ipynb
├── dataset/
│   └── MIMII/               # MIMII dataset (fan, pump, slider/Slide Rail, valve)
├── results/
│   ├── splits/              # train.csv, validation.csv, test.csv
│   ├── models/              # acfa_cnn_best.pth, resnet18_best.pth
│   ├── figures/             # Confusion matrices, training curves, spectrograms
│   │   └── explainability/  # Grad-CAM activation heatmaps
│   └── metrics/             # Classification reports, JSON metrics, model comparison
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── .gitignore
```

## Machine Categories & Classification Classes

Supported machine categories (4) and exact classification classes (8):
1. `Fan_Normal` & `Fan_Anomaly`
2. `Pump_Normal` & `Pump_Anomaly`
3. `Valve_Normal` & `Valve_Anomaly`
4. `SlideRail_Normal` & `SlideRail_Anomaly`

## Preprocessing Pipeline

Canonical 5-step audio processing pipeline:
1. **Load:** `.wav` audio at 16,000 Hz.
2. **Channel Downmix:** 8-channel array mixed to 1 mono channel (`mean`).
3. **Amplitude Normalization:** Peak normalized to `[-1.0, 1.0]`.
4. **Segmentation:** Fixed 2-second windows with 1-second hop.
5. **Mel-Spectrogram:** 64-bin Log-Mel Spectrogram computation `(64, 61)`.

## Model Performance & Evaluation Metrics

Evaluated on the untouched 60-sample test set (`test.csv`):

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline CNN** | 13.33% | 1.67% | 12.50% | 2.94% | 3.14% |
| **ResNet18 Transfer Learning** | **76.67%** | **83.67%** | **76.56%** | **76.05%** | **75.93%** |

*Artifacts:*
- Baseline CNN Confusion Matrix: [`results/figures/cnn_confusion_matrix.png`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/figures/cnn_confusion_matrix.png)
- ResNet18 Confusion Matrix: [`results/figures/resnet18_confusion_matrix.png`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/figures/resnet18_confusion_matrix.png)
- Model Comparison Plot: [`results/figures/model_comparison.png`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/figures/model_comparison.png)
- Model Comparison Table: [`results/metrics/model_comparison.csv`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/metrics/model_comparison.csv)

## Explainability & Grad-CAM (Milestone 7)

Grad-CAM class activation heatmaps are generated targeting `layer4` of ResNet18 to visualize spectral/temporal frequency regions contributing to model predictions.

*Artifacts saved under:* [`results/figures/explainability/`](file:///c:/Users/akshitha/Downloads/ACFA-DL-20260922T130546Z-1-001/ACFA-DL/results/figures/explainability)
> *Note:* Grad-CAM visualizes neural network feature activation regions and does not constitute physical diagnostic root causes.

## How to Run

1. **Navigate into the project directory:**
   ```bash
   cd ACFA-DL
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Baseline CNN Evaluation (M5):**
   ```bash
   python src/evaluation.py
   ```

4. **Train & Evaluate ResNet18 Transfer Learning (M6):**
   ```bash
   python src/transfer_learning.py
   ```

5. **Generate Explainability & Grad-CAM Visualizations (M7):**
   ```bash
   python src/visualization.py
   ```

6. **Launch Interactive Streamlit Web Application (M8):**
   ```bash
   python -m streamlit run app/streamlit_app.py
   ```


