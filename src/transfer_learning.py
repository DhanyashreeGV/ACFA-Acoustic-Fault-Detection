import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.models as models

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import CLASSES, CLASS_TO_IDX
from src.dataset import create_dataloader
from src.evaluation import evaluate_model_pipeline, plot_confusion_matrix

class ResNet18Transfer(nn.Module):
    """
    ResNet18 Transfer Learning Model for Acoustic Fault Classification.
    Expands 1-channel Mel-Spectrogram inputs (B, 1, 64, 61) to 3 channels (B, 3, 64, 61).
    Replaces FC head to output 8 classes.
    """
    def __init__(self, num_classes: int = 8, freeze_backbone: bool = False, pretrained: bool = False):
        super(ResNet18Transfer, self).__init__()
        
        # Load ResNet18 architecture
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                self.resnet = models.resnet18(weights=weights)
            except Exception:
                self.resnet = models.resnet18(weights=None)
        else:
            self.resnet = models.resnet18(weights=None)
            
        if freeze_backbone and pretrained:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        # Replace final classification layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def unfreeze_layer4(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, 1, 64, 61) or (B, 64, 61)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # Expand 1 channel -> 3 channels
            
        return self.resnet(x)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

def train_and_evaluate_resnet18(p_root: str, epochs: int = 10, batch_size: int = 16):
    print("\n==================================================")
    print("RUNNING RESNET18 TRANSFER LEARNING & EVALUATION (M6)")
    print("==================================================")
    
    train_csv = os.path.join(p_root, 'results', 'splits', 'train.csv')
    val_csv = os.path.join(p_root, 'results', 'splits', 'validation.csv')
    test_csv = os.path.join(p_root, 'results', 'splits', 'test.csv')
    models_dir = os.path.join(p_root, 'results', 'models')
    metrics_dir = os.path.join(p_root, 'results', 'metrics')
    fig_dir = os.path.join(p_root, 'results', 'figures')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    device = 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Dataloaders
    train_loader = create_dataloader(train_csv, base_dir=p_root, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_csv, base_dir=p_root, batch_size=batch_size, shuffle=False)
    
    model = ResNet18Transfer(num_classes=8, freeze_backbone=False, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_ckpt_path = os.path.join(models_dir, 'resnet18_best.pth')
    
    print("\n--- Training ResNet18 Architecture ---")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.2f}%")
        
        if v_acc >= best_val_acc or epoch == 1:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_ckpt_path)
            
    print(f"\nBest Validation Accuracy achieved: {best_val_acc * 100:.2f}%")
    print(f"Saved best ResNet18 checkpoint to '{best_ckpt_path}'")
    
    # Save Training History JSON
    history_path = os.path.join(metrics_dir, 'resnet18_training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to '{history_path}'")
    
    # Save Training Curves Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep_range = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(ep_range, history['train_loss'], label='Train Loss', color='blue', marker='o')
    ax1.plot(ep_range, history['val_loss'], label='Val Loss', color='orange', marker='s')
    ax1.set_title('ResNet18 Loss Curves', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CrossEntropy Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    ax2.plot(ep_range, [a * 100 for a in history['train_acc']], label='Train Acc', color='blue', marker='o')
    ax2.plot(ep_range, [a * 100 for a in history['val_acc']], label='Val Acc', color='orange', marker='s')
    ax2.set_title('ResNet18 Accuracy Curves', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    curves_path = os.path.join(fig_dir, 'resnet18_training_curves.png')
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"Saved training curves to '{curves_path}'")
    
    # EVALUATE ON UNTOUCHED TEST SET
    print("\n--- EVALUATING RESNET18 ON TEST SET ---")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
    cm, class_metrics, overall_metrics, y_true, y_pred = evaluate_model_pipeline(
        model=model,
        test_csv_path=test_csv,
        base_dir=p_root,
        device=device
    )
    
    # Save ResNet18 Reports
    df_report = pd.DataFrame(class_metrics)
    df_report.to_csv(os.path.join(metrics_dir, 'resnet18_classification_report.csv'), index=False)
    
    resnet_metrics_path = os.path.join(metrics_dir, 'resnet18_metrics.json')
    with open(resnet_metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'model_name': 'ResNet18_Transfer', 'overall_metrics': overall_metrics, 'per_class_metrics': class_metrics}, f, indent=2)
        
    plot_confusion_matrix(cm, CLASSES, os.path.join(fig_dir, 'resnet18_confusion_matrix.png'), title="ResNet18 Confusion Matrix")
    
    # MODEL COMPARISON (Baseline CNN vs ResNet18)
    cnn_metrics_path = os.path.join(metrics_dir, 'cnn_metrics.json')
    with open(cnn_metrics_path, 'r', encoding='utf-8') as f:
        cnn_json = json.load(f)['overall_metrics']
        
    comparison_data = [
        {
            'Model': 'Baseline CNN',
            'Accuracy': cnn_json['accuracy'],
            'Macro Precision': cnn_json['macro_precision'],
            'Macro Recall': cnn_json['macro_recall'],
            'Macro F1': cnn_json['macro_f1'],
            'Weighted F1': cnn_json['weighted_f1']
        },
        {
            'Model': 'ResNet18 Transfer',
            'Accuracy': overall_metrics['accuracy'],
            'Macro Precision': overall_metrics['macro_precision'],
            'Macro Recall': overall_metrics['macro_recall'],
            'Macro F1': overall_metrics['macro_f1'],
            'Weighted F1': overall_metrics['weighted_f1']
        }
    ]
    
    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv(os.path.join(metrics_dir, 'model_comparison.csv'), index=False)
    print("\n=== MODEL COMPARISON TABLE ===")
    print(df_comp.to_string(index=False))
    
    # Plot Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_to_plot = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Weighted F1']
    x = np.arange(len(metrics_to_plot))
    w = 0.35
    
    cnn_vals = [df_comp.loc[0, m] * 100 for m in metrics_to_plot]
    resnet_vals = [df_comp.loc[1, m] * 100 for m in metrics_to_plot]
    
    ax.bar(x - w/2, cnn_vals, w, label='Baseline CNN', color='#d62728')
    ax.bar(x + w/2, resnet_vals, w, label='ResNet18 Transfer', color='#2ca02c')
    
    ax.set_title('Baseline CNN vs ResNet18 Transfer Learning Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    print("Saved model comparison plot to 'results/figures/model_comparison.png'")
    
    return overall_metrics

if __name__ == '__main__':
    train_and_evaluate_resnet18(project_root)
