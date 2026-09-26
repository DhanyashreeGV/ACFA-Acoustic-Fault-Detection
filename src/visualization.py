import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import CLASSES, load_wav_file, resolve_filepath
from src.preprocessing import process_file, mix_channels, normalize_amplitude, compute_mel_spectrogram

def predict_audio(filepath: str, model, device: str = 'cpu'):
    """
    Run canonical preprocessing and model inference on a single audio file.
    Returns dict with predicted_machine, predicted_condition, predicted_class, confidence, probabilities.
    """
    model.eval()
    model.to(device)
    
    # Preprocess
    specs = process_file(filepath)
    spec = specs[len(specs) // 2]
    
    # Input tensor (1, 1, 64, 61)
    tensor_spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor_spec)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        
    pred_class = CLASSES[pred_idx]
    
    # Parse machine & condition
    parts = pred_class.split('_')
    machine = parts[0]
    condition = parts[1]
    
    return {
        'filepath': filepath,
        'predicted_machine': machine,
        'predicted_condition': condition,
        'predicted_class': pred_class,
        'confidence': confidence,
        'probabilities': {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        'spectrogram': spec
    }

class GradCAM:
    """
    Grad-CAM Implementation for ResNet18 / CNN Models.
    Computes class activation heatmaps showing regions contributing to model prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        score = output[0, class_idx]
        score.backward()
        
        # Global Average Pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (1, 1, H, W)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam

def plot_prediction_summary(
    filepath: str,
    model,
    output_path: str = None,
    device: str = 'cpu'
):
    """
    Generate audio signal waveform, Mel-spectrogram, and prediction summary card.
    """
    pred_res = predict_audio(filepath, model, device=device)
    sr, raw = load_wav_file(filepath)
    mono = mix_channels(raw, mode='mean')
    norm = normalize_amplitude(mono, target_peak=1.0)
    spec = pred_res['spectrogram']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    # 1. Raw Waveform
    time_axis = np.linspace(0, len(norm)/sr, len(norm))
    axes[0, 0].plot(time_axis, norm, color='navy', alpha=0.75, linewidth=0.6)
    axes[0, 0].set_title("Normalized Time-Domain Waveform", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Time (seconds)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, linestyle='--', alpha=0.4)
    
    # 2. Log-Mel Spectrogram
    im = axes[0, 1].imshow(spec, aspect='auto', origin='lower', cmap='magma')
    axes[0, 1].set_title("64-Bin Log-Mel Spectrogram", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Time Frames")
    axes[0, 1].set_ylabel("Mel Frequency Bins")
    fig.colorbar(im, ax=axes[0, 1], format='%+2.0f dB')
    
    # 3. Class Probabilities Bar Chart
    classes = list(pred_res['probabilities'].keys())
    probs = [pred_res['probabilities'][c] * 100 for c in classes]
    colors = ['crimson' if c == pred_res['predicted_class'] else 'steelblue' for c in classes]
    
    axes[1, 0].barh(classes, probs, color=colors)
    axes[1, 0].set_title("Model Prediction Class Probabilities", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Probability (%)")
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].grid(True, linestyle='--', alpha=0.4)
    
    # 4. Summary Text Card
    axes[1, 1].axis('off')
    summary_text = (
        f"--- PREDICTION SUMMARY CARD ---\n\n"
        f"File: {os.path.basename(filepath)}\n"
        f"Sampling Rate: {sr} Hz\n"
        f"Audio Duration: {len(raw)/sr:.1f} s\n\n"
        f"Predicted Machine   : {pred_res['predicted_machine']}\n"
        f"Predicted Condition : {pred_res['predicted_condition']}\n"
        f"Predicted Class     : {pred_res['predicted_class']}\n"
        f"Confidence Score    : {pred_res['confidence']*100:.2f}%\n"
    )
    axes[1, 1].text(0.1, 0.2, summary_text, fontsize=12, family='monospace',
                     bbox=dict(boxstyle="round,pad=1.0", facecolor="aliceblue", edgecolor="dodgerblue", lw=2))
    
    plt.suptitle(f"ACFA Acoustic Fault Analyzer — Prediction & Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved prediction summary to '{output_path}'")
        
    return pred_res

def generate_gradcam_visualization(
    filepath: str,
    model,
    target_layer,
    output_path: str,
    device: str = 'cpu'
):
    """
    Generate and save Grad-CAM explainability heatmap overlaid on Mel-Spectrogram.
    """
    grad_cam = GradCAM(model, target_layer)
    specs = process_file(filepath)
    spec = specs[len(specs) // 2]  # Shape: (64, 61)
    
    tensor_spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    tensor_spec.requires_grad = True
    
    # Model prediction
    output = model(tensor_spec)
    probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    
    heatmap = grad_cam.generate_heatmap(tensor_spec, class_idx=pred_idx)
    
    # Plot Grad-CAM overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Mel Spectrogram
    axes[0].imshow(spec, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title("Log-Mel Spectrogram", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Time Frames")
    axes[0].set_ylabel("Mel Frequency Bins")
    
    # Grad-CAM Heatmap
    axes[1].imshow(heatmap, aspect='auto', origin='lower', cmap='jet')
    axes[1].set_title(f"Grad-CAM Heatmap ({pred_class})", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Time Frames")
    
    # Overlaid View
    axes[2].imshow(spec, aspect='auto', origin='lower', cmap='gray')
    axes[2].imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
    axes[2].set_title("Overlaid Activation Map", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Time Frames")
    
    disclaimer = "Note: Grad-CAM highlights spectral regions contributing to neural network predictions (feature activations)."
    plt.suptitle(f"ACFA Explainability — {pred_class} (Confidence: {probs[pred_idx]*100:.1f}%)\n{disclaimer}", fontsize=11, fontweight='bold', y=1.03)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Grad-CAM visualization to '{output_path}'")

def run_explainability_demo(p_root: str):
    """Generate example explainability visualizations for all 4 machine categories."""
    print("\n==================================================")
    print("RUNNING EXPLAINABILITY & VISUALIZATION (M7)")
    print("==================================================")
    
    from src.transfer_learning import ResNet18Transfer
    
    ckpt_path = os.path.join(p_root, 'results', 'models', 'resnet18_best.pth')
    exp_dir = os.path.join(p_root, 'results', 'figures', 'explainability')
    os.makedirs(exp_dir, exist_ok=True)
    
    model = ResNet18Transfer(num_classes=8)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.eval()
    
    target_samples = {
        'Fan': os.path.join(p_root, 'dataset', 'MIMII', 'fan', 'abnormal', '00000000.wav'),
        'Pump': os.path.join(p_root, 'dataset', 'MIMII', 'pump', 'abnormal', '00000000.wav'),
        'Valve': os.path.join(p_root, 'dataset', 'MIMII', 'valve', 'abnormal', '00000000.wav'),
        'SlideRail': os.path.join(p_root, 'dataset', 'MIMII', 'slider', 'abnormal', '00000000.wav')
    }
    
    for m_name, wav_path in target_samples.items():
        summary_out = os.path.join(exp_dir, f'prediction_summary_{m_name.lower()}.png')
        plot_prediction_summary(wav_path, model, output_path=summary_out, device='cpu')
        
        gradcam_out = os.path.join(exp_dir, f'gradcam_{m_name.lower()}.png')
        generate_gradcam_visualization(wav_path, model, target_layer=model.resnet.layer4[-1], output_path=gradcam_out, device='cpu')

if __name__ == '__main__':
    run_explainability_demo(project_root)
