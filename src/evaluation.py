import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import CLASSES, CLASS_TO_IDX, resolve_filepath, standardize_label
from src.preprocessing import process_file
from src.models import load_acfa_model

def calculate_confusion_matrix(y_true: list, y_pred: list, num_classes: int = 8) -> np.ndarray:
    """Compute (num_classes, num_classes) confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def calculate_class_metrics(cm: np.ndarray):
    """Compute per-class precision, recall, f1-score, support."""
    num_classes = cm.shape[0]
    metrics = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics.append({
            'class': CLASSES[i],
            'precision': float(precision),
            'recall': float(recall),
            'f1-score': float(f1),
            'support': int(support)
        })
        
    return metrics

def calculate_overall_metrics(cm: np.ndarray, class_metrics: list):
    """Compute overall accuracy, macro, and weighted metrics."""
    total_samples = cm.sum()
    correct_samples = np.trace(cm)
    accuracy = float(correct_samples / total_samples) if total_samples > 0 else 0.0
    
    supports = [m['support'] for m in class_metrics]
    precisions = [m['precision'] for m in class_metrics]
    recalls = [m['recall'] for m in class_metrics]
    f1s = [m['f1-score'] for m in class_metrics]
    
    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    
    sum_supports = sum(supports)
    if sum_supports > 0:
        weighted_prec = float(np.average(precisions, weights=supports))
        weighted_rec = float(np.average(recalls, weights=supports))
        weighted_f1 = float(np.average(f1s, weights=supports))
    else:
        weighted_prec = 0.0
        weighted_rec = 0.0
        weighted_f1 = 0.0
    
    return {
        'total_samples': int(total_samples),
        'correct_predictions': int(correct_samples),
        'accuracy': accuracy,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_prec,
        'weighted_recall': weighted_rec,
        'weighted_f1': weighted_f1
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: str,
    title: str = "CNN Confusion Matrix"
):
    """Save clearly labelled confusion matrix heatmap figure."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Text annotations inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
            
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to '{output_path}'")

def evaluate_model_pipeline(
    model,
    test_csv_path: str,
    base_dir: str = None,
    device: str = 'cpu'
):
    """
    Perform full test dataset inference and evaluation pipeline.
    """
    model.eval()
    model.to(device)
    
    df = pd.read_csv(test_csv_path)
    y_true = []
    y_pred = []
    y_probs = []
    
    failed_samples = 0
    
    for _, row in df.iterrows():
        raw_path = str(row['filepath'])
        local_path = resolve_filepath(raw_path, base_dir=base_dir)
        std_label = standardize_label(str(row['label']))
        true_idx = CLASS_TO_IDX[std_label]
        
        try:
            specs = process_file(local_path)
            spec = specs[len(specs) // 2]
            
            # Format input tensor (1, 1, 64, 61)
            tensor_spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(tensor_spec)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_probs.append(probs.tolist())
        except Exception as e:
            print(f"Error processing '{local_path}': {e}")
            failed_samples += 1

    if len(y_true) == 0:
        print("\n" + "="*70)
        print("ERROR: 0 audio files were processed! All audio files failed to load.")
        print("Please ensure your dataset files (.wav) exist in the 'dataset/MIMII/' folder.")
        print("If on Google Colab, upload 'dataset.zip' and unzip it:")
        print("  !unzip -o dataset.zip -d /content/ACFA-Acoustic-Fault-Detection/")
        print("="*70 + "\n")

    cm = calculate_confusion_matrix(y_true, y_pred, num_classes=len(CLASSES))
    per_class_metrics = calculate_class_metrics(cm)
    overall_metrics = calculate_overall_metrics(cm, per_class_metrics)
    overall_metrics['failed_samples'] = failed_samples
    
    return cm, per_class_metrics, overall_metrics, y_true, y_pred


def run_baseline_cnn_evaluation(p_root: str):
    """Main function to run Baseline CNN evaluation."""
    print("\n==================================================")
    print("RUNNING BASELINE CNN EVALUATION (M5)")
    print("==================================================")
    
    ckpt_path = os.path.join(p_root, 'results', 'models', 'acfa_cnn_best.pth')
    test_csv = os.path.join(p_root, 'results', 'splits', 'test.csv')
    
    model = load_acfa_model(ckpt_path, device='cpu')
    cm, class_metrics, overall_metrics, y_true, y_pred = evaluate_model_pipeline(
        model=model,
        test_csv_path=test_csv,
        base_dir=p_root,
        device='cpu'
    )
    
    metrics_dir = os.path.join(p_root, 'results', 'metrics')
    fig_dir = os.path.join(p_root, 'results', 'figures')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Save Classification Report CSV
    df_report = pd.DataFrame(class_metrics)
    csv_report_path = os.path.join(metrics_dir, 'cnn_classification_report.csv')
    df_report.to_csv(csv_report_path, index=False)
    print(f"Saved classification report to '{csv_report_path}'")
    
    # Save Metrics JSON
    json_metrics_path = os.path.join(metrics_dir, 'cnn_metrics.json')
    full_metrics_payload = {
        'model_name': 'Baseline_CNN',
        'overall_metrics': overall_metrics,
        'per_class_metrics': class_metrics
    }
    with open(json_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(full_metrics_payload, f, indent=2)
    print(f"Saved metrics JSON to '{json_metrics_path}'")
    
    # Save Confusion Matrix Plot
    cm_plot_path = os.path.join(fig_dir, 'cnn_confusion_matrix.png')
    plot_confusion_matrix(cm, CLASSES, cm_plot_path, title="Baseline CNN Confusion Matrix")
    
    print("\n--- BASELINE CNN EVALUATION RESULTS ---")
    print(f"Test Samples evaluated: {overall_metrics['total_samples']}")
    print(f"Accuracy          : {overall_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision   : {overall_metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall      : {overall_metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score    : {overall_metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1-Score : {overall_metrics['weighted_f1'] * 100:.2f}%")
    
    return overall_metrics

if __name__ == '__main__':
    run_baseline_cnn_evaluation(project_root)
