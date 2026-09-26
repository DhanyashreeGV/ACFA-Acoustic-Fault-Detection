import os
import sys

# Windows DLL handling for PyTorch environment compatibility
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
for p in [r'C:\Users\akshitha\anaconda3\Library\bin', r'C:\Users\akshitha\anaconda3\Lib\site-packages\torch\lib']:
    if os.path.exists(p) and hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(p)
        except Exception:
            pass

import tempfile
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import CLASSES, load_wav_file, resolve_filepath
from src.preprocessing import process_file, mix_channels, normalize_amplitude, compute_mel_spectrogram
from src.models import load_acfa_model
from src.transfer_learning import ResNet18Transfer
from src.visualization import GradCAM

# Page Configuration
st.set_page_config(
    page_title="ACFA — Acoustic Fault Analyzer",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.3rem; color: #1E3A8A; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #2563EB; }
    .stAlert { border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔊 ACFA — Acoustic Fault Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Acoustic Fault Detection and Classification Using Deep Learning (MIMII Dataset)</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("⚙️ Model & Settings")
model_choice = st.sidebar.radio(
    "Select Classifier Model:",
    ["ResNet18 Transfer Learning (Recommended)", "Baseline CNN"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Supported Classes (8)")
for cls_name in CLASSES:
    st.sidebar.markdown(f"- `{cls_name}`")

@st.cache_resource
def get_model(choice: str):
    """Cached loader for trained model checkpoints."""
    device = 'cpu'
    if "ResNet18" in choice:
        ckpt_p = os.path.join(project_root, 'results', 'models', 'resnet18_best.pth')
        m = ResNet18Transfer(num_classes=8)
        if os.path.exists(ckpt_p):
            m.load_state_dict(torch.load(ckpt_p, map_location=device, weights_only=True))
        m.eval()
        return m, "ResNet18 Transfer Learning"
    else:
        ckpt_p = os.path.join(project_root, 'results', 'models', 'acfa_cnn_best.pth')
        m = load_acfa_model(ckpt_p, device=device)
        return m, "Baseline CNN"

try:
    active_model, model_name_str = get_model(model_choice)
    st.sidebar.success(f"Loaded {model_name_str} successfully!")
except Exception as e:
    st.sidebar.error(f"Failed to load model checkpoint: {e}")
    st.stop()

# File Upload / Sample Selection
st.subheader("📁 Audio File Input")
col_up, col_sample = st.columns([2, 1])

with col_up:
    uploaded_file = st.file_uploader("Upload an industrial WAV audio file (16 kHz):", type=["wav"])

with col_sample:
    sample_choice = st.selectbox(
        "Or pick a sample from dataset:",
        ["None", "Fan Normal", "Fan Anomaly", "Pump Normal", "Pump Anomaly", "Valve Normal", "Valve Anomaly", "Slide Rail Normal", "Slide Rail Anomaly"]
    )

sample_path_map = {
    "Fan Normal": resolve_filepath("dataset/MIMII/fan/normal/00000000.wav", base_dir=project_root),
    "Fan Anomaly": resolve_filepath("dataset/MIMII/fan/abnormal/00000000.wav", base_dir=project_root),
    "Pump Normal": resolve_filepath("dataset/MIMII/pump/normal/00000000.wav", base_dir=project_root),
    "Pump Anomaly": resolve_filepath("dataset/MIMII/pump/abnormal/00000000.wav", base_dir=project_root),
    "Valve Normal": resolve_filepath("dataset/MIMII/valve/normal/00000000.wav", base_dir=project_root),
    "Valve Anomaly": resolve_filepath("dataset/MIMII/valve/abnormal/00000000.wav", base_dir=project_root),
    "Slide Rail Normal": resolve_filepath("dataset/MIMII/slider/normal/00000000.wav", base_dir=project_root),
    "Slide Rail Anomaly": resolve_filepath("dataset/MIMII/slider/abnormal/00000000.wav", base_dir=project_root)
}

target_filepath = None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        target_filepath = tmp.name
elif sample_choice != "None":
    target_filepath = sample_path_map[sample_choice]


if target_filepath is None:
    st.info(" Please upload a `.wav` file or select a sample from the dropdown menu to begin analysis.")
    st.stop()

# Audio Validation & Preprocessing
try:
    sr, raw_audio = load_wav_file(target_filepath)
    mono_audio = mix_channels(raw_audio, mode='mean')
    norm_audio = normalize_amplitude(mono_audio, target_peak=1.0)
    spectrograms = process_file(target_filepath)
    spec = spectrograms[len(spectrograms) // 2]
except Exception as err:
    st.error(f"Failed to process audio file. Please ensure it is a valid 16 kHz WAV audio file.\nDetails: {err}")
    st.stop()

# Audio Information Banner
num_channels = 1 if raw_audio.ndim == 1 else raw_audio.shape[1]
duration_sec = len(raw_audio) / float(sr)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sampling Rate", f"{sr} Hz")
c2.metric("Channels", f"{num_channels} ({'Array' if num_channels > 1 else 'Mono'})")
c3.metric("Duration", f"{duration_sec:.1f} sec")
c4.metric("Active Model", model_name_str.split()[0])

st.markdown("---")

# Model Inference
tensor_spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    logits = active_model(tensor_spec)
    probs = F.softmax(logits, dim=1).numpy()[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

pred_class = CLASSES[pred_idx]
machine_type, condition = pred_class.split('_')

# Main Analysis Tabs
tab_pred, tab_sig, tab_explain = st.tabs(["🎯 Classification Results", "📈 Audio Signal & Spectrogram", "🔍 Grad-CAM Explainability"])

with tab_pred:
    st.subheader("Model Diagnostic Prediction")
    
    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
    with col_res1:
        st.markdown(f"**Predicted Machine**\n### {machine_type}")
    with col_res2:
        cond_color = "🟢" if condition == "Normal" else "🔴"
        st.markdown(f"**Operating Condition**\n### {cond_color} {condition}")
    with col_res3:
        st.markdown(f"**Full Class Label**\n### `{pred_class}`")
    with col_res4:
        st.markdown(f"**Confidence Score**\n### {confidence*100:.1f}%")
        
    st.markdown("---")
    st.subheader("Class Probability Distribution")
    
    df_probs = pd.DataFrame({
        'Class': CLASSES,
        'Probability (%)': [p * 100 for p in probs]
    }).sort_values(by='Probability (%)', ascending=True)
    
    fig_prob, ax_prob = plt.subplots(figsize=(10, 4))
    bars = ax_prob.barh(df_probs['Class'], df_probs['Probability (%)'], color=['#2563EB' if c == pred_class else '#9CA3AF' for c in df_probs['Class']])
    ax_prob.set_xlabel("Probability (%)", fontsize=11)
    ax_prob.set_xlim(0, 100)
    ax_prob.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig_prob)

with tab_sig:
    col_wave, col_spec = st.columns(2)
    
    with col_wave:
        st.subheader("Time-Domain Waveform")
        fig_w, ax_w = plt.subplots(figsize=(6, 4))
        time_ax = np.linspace(0, duration_sec, len(norm_audio))
        ax_w.plot(time_ax, norm_audio, color='#1E3A8A', alpha=0.75, linewidth=0.6)
        ax_w.set_xlabel("Time (seconds)")
        ax_w.set_ylabel("Normalized Amplitude")
        ax_w.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig_w)
        
    with col_spec:
        st.subheader("64-Bin Log-Mel Spectrogram")
        fig_s, ax_s = plt.subplots(figsize=(6, 4))
        im_s = ax_s.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        ax_s.set_xlabel("Time Frames")
        ax_s.set_ylabel("Mel Frequency Bins")
        fig_s.colorbar(im_s, ax=ax_s, format='%+2.0f dB')
        st.pyplot(fig_s)

with tab_explain:
    st.subheader("Grad-CAM Saliency Activation Heatmap")
    
    if "ResNet18" in model_name_str:
        try:
            tensor_spec_grad = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            tensor_spec_grad.requires_grad = True
            
            grad_cam = GradCAM(active_model, active_model.resnet.layer4[-1])
            heatmap = grad_cam.generate_heatmap(tensor_spec_grad, class_idx=pred_idx)
            
            fig_g, axes_g = plt.subplots(1, 3, figsize=(14, 4))
            axes_g[0].imshow(spec, aspect='auto', origin='lower', cmap='magma')
            axes_g[0].set_title("Log-Mel Spectrogram", fontsize=10, fontweight='bold')
            
            axes_g[1].imshow(heatmap, aspect='auto', origin='lower', cmap='jet')
            axes_g[1].set_title("Grad-CAM Activation Map", fontsize=10, fontweight='bold')
            
            axes_g[2].imshow(spec, aspect='auto', origin='lower', cmap='gray')
            axes_g[2].imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
            axes_g[2].set_title("Overlaid Activation View", fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig_g)
            
            st.info("💡 **Explainability Note:** Grad-CAM heatmaps highlight spectral and temporal regions that strongly contribute to the neural network's activation score for the predicted class. They do not constitute physical diagnostic root causes.")
        except Exception as e_cam:
            st.warning(f"Grad-CAM visualization error: {e_cam}")
    else:
        st.info("Grad-CAM feature activation maps are enabled for the ResNet18 Transfer Learning model. Please select ResNet18 in the sidebar to view Grad-CAM heatmaps.")

# Clean up temp file if created
if uploaded_file is not None and target_filepath and os.path.exists(target_filepath):
    try:
        os.unlink(target_filepath)
    except Exception:
        pass
