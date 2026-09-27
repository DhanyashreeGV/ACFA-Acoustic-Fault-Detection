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
from src.preprocessing import process_file, mix_channels, normalize_amplitude
from src.models import load_acfa_model
from src.transfer_learning import ResNet18Transfer
from src.visualization import GradCAM

# Page Configuration
st.set_page_config(
    page_title="ACFA — Industrial Acoustic Diagnostic System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Top Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.55;
    }
    
    /* Cards & Containers */
    .card-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    .step-badge {
        background: #2563eb;
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        margin-right: 0.4rem;
    }

    /* Diagnosis Status Banners */
    .status-box-normal {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 0.85rem;
        padding: 1.5rem;
        color: #14532d;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.12);
    }
    .status-box-anomaly {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 0.85rem;
        padding: 1.5rem;
        color: #7f1d1d;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.12);
    }
    
    .pill-label {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .pill-normal { background: #bbf7d0; color: #166534; }
    .pill-anomaly { background: #fecaca; color: #991b1b; }
    
    /* Action Box */
    .action-box {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 0.9rem 1.2rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin-top: 0.8rem;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# Top Hero Header
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">AI Machinery Health Inspector</div>
    <div class="hero-title">🏭 ACFA Acoustic Fault Analyzer</div>
    <div class="hero-subtitle">
        Listen to the sound of industrial machinery and instantly detect abnormal mechanical behavior (bearing wear, cavitation, rail friction, or leakage) before catastrophic equipment failure occurs.
    </div>
</div>
""", unsafe_allow_html=True)

# First-time User Guide (Always clearly visible in an attractive card)
with st.expander("📖 **First time using ACFA? Click here for the 60-Second Quick Start Guide**", expanded=False):
    st.markdown("""
    <div class="card-box" style="background:#f8fafc; margin-bottom:0;">
        <h4 style="margin-top:0; color:#0f172a;">How to use this tool in 3 simple steps:</h4>
        <div style="display:flex; gap:1.5rem; flex-wrap:wrap; margin-top:0.8rem;">
            <div style="flex:1; min-width:240px;">
                <span class="step-badge">1</span> <b>Pick or Upload Machine Audio</b><br>
                <span style="font-size:0.9rem; color:#475569;">
                    Click any of the <b>Quick-Demo buttons</b> below to test a healthy vs faulty machine instantly, or upload a custom <code>.wav</code> recording.
                </span>
            </div>
            <div style="flex:1; min-width:240px;">
                <span class="step-badge">2</span> <b>Listen to the Sound</b><br>
                <span style="font-size:0.9rem; color:#475569;">
                    Press play on the audio player. Compare the regular hum of healthy equipment against the rattling/hissing of a damaged machine.
                </span>
            </div>
            <div style="flex:1; min-width:240px;">
                <span class="step-badge">3</span> <b>Read AI Diagnosis & Insights</b><br>
                <span style="font-size:0.9rem; color:#475569;">
                    Review the <b>Health Status Card</b>, confidence probability, and inspect <b>Grad-CAM heatmaps</b> showing which sound pitches triggered the alert.
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ System Configuration")
model_choice = st.sidebar.radio(
    "Deep Learning Model:",
    ["ResNet18 Transfer Learning (Recommended)", "Baseline CNN"],
    index=0,
    help="ResNet18 transfer learning achieves 76.7% test accuracy and enables Grad-CAM explainability heatmaps."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏭 Equipment Types Monitored")
st.sidebar.markdown("""
- **🌀 Industrial Fan**: Detects worn motor bearings & loose blades.
- **🚰 Water Pump**: Detects impeller cavitation & air intake.
- **🎚️ Flow Valve**: Detects pressure seal leaks & flow resistance.
- **🛤️ Linear Slide Rail**: Detects rail guide friction & track obstruction.
""")

st.sidebar.markdown("---")
with st.sidebar.expander("📚 **Glossary: Understand Technical Terms**"):
    st.markdown("""
    - **Log-Mel Spectrogram**: A visual 'fingerprint' of sound. Time is left-to-right, frequency (pitch) is bottom-to-top, and brightness is volume.
    - **Grad-CAM**: Gradient-weighted Class Activation Mapping. It shows the neural network's 'focus area' by highlighting suspicious sound frequencies in red/yellow.
    - **Cavitation**: Bubbles collapsing violently inside a pump, producing characteristic high-frequency crackling sounds.
    - **Confidence Score**: The probability (0–100%) that the AI's diagnosis is correct.
    """)

st.sidebar.markdown("""
<div style="font-size:0.8rem; color:#94a3b8; margin-top:1.5rem; line-height:1.4;">
    <b>Benchmark:</b> MIMII Dataset<br>
    <b>Sampling Rate:</b> 16,000 Hz Mono<br>
    <b>Feature Resolution:</b> 64 Mel Bins
</div>
""", unsafe_allow_html=True)

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
    st.sidebar.success(f"🟢 Active: **{model_name_str}**")
except Exception as e:
    st.sidebar.error(f"Failed to load model checkpoint: {e}")
    st.stop()

# Preset Sample Map
sample_presets = {
    "fan_normal": {
        "name": "🌀 Fan — Normal Operation",
        "desc": "Smooth, steady motor rotation with uniform air humming.",
        "path": resolve_filepath("dataset/MIMII/fan/normal/00000000.wav", base_dir=project_root)
    },
    "fan_abnormal": {
        "name": "🌀 Fan — Bearing Fault (Anomaly)",
        "desc": "Mechanical rattling and vibration caused by worn motor bearings.",
        "path": resolve_filepath("dataset/MIMII/fan/abnormal/00000000.wav", base_dir=project_root)
    },
    "pump_normal": {
        "name": "🚰 Pump — Normal Operation",
        "desc": "Consistent hydraulic hum with steady fluid circulation.",
        "path": resolve_filepath("dataset/MIMII/pump/normal/00000000.wav", base_dir=project_root)
    },
    "pump_abnormal": {
        "name": "🚰 Pump — Cavitation (Anomaly)",
        "desc": "Erratic popping sound caused by vapor bubble collapse inside pump chamber.",
        "path": resolve_filepath("dataset/MIMII/pump/abnormal/00000000.wav", base_dir=project_root)
    },
    "valve_normal": {
        "name": "🎚️ Valve — Normal Operation",
        "desc": "Clean, periodic opening and closing sound without hiss.",
        "path": resolve_filepath("dataset/MIMII/valve/normal/00000000.wav", base_dir=project_root)
    },
    "valve_abnormal": {
        "name": "🎚️ Valve — Gas/Fluid Leak (Anomaly)",
        "desc": "High-pitched hissing sound caused by defective sealing gasket.",
        "path": resolve_filepath("dataset/MIMII/valve/abnormal/00000000.wav", base_dir=project_root)
    },
    "slider_normal": {
        "name": "🛤️ Slide Rail — Normal Operation",
        "desc": "Smooth linear gliding motion along clean precision rail.",
        "path": resolve_filepath("dataset/MIMII/slider/normal/00000000.wav", base_dir=project_root)
    },
    "slider_abnormal": {
        "name": "🛤️ Slide Rail — Obstruction / Friction (Anomaly)",
        "desc": "Friction scraping sound caused by rail debris or lack of lubrication.",
        "path": resolve_filepath("dataset/MIMII/slider/abnormal/00000000.wav", base_dir=project_root)
    }
}

# Session State for Selected Audio
if "selected_sample_key" not in st.session_state:
    st.session_state.selected_sample_key = "fan_normal"

st.markdown("### 🎧 Step 1: Select Audio Sample")

# One-Click Quick Test Buttons
st.markdown("##### ⚡ Quick 1-Click Test Scenarios:")
btn_c1, btn_c2, btn_c3, btn_c4, btn_c5 = st.columns(5)

with btn_c1:
    if st.button("🌀 Normal Fan", use_container_width=True):
        st.session_state.selected_sample_key = "fan_normal"
with btn_c2:
    if st.button("⚠️ Faulty Fan", use_container_width=True):
        st.session_state.selected_sample_key = "fan_abnormal"
with btn_c3:
    if st.button("⚠️ Pump Cavitation", use_container_width=True):
        st.session_state.selected_sample_key = "pump_abnormal"
with btn_c4:
    if st.button("⚠️ Leaking Valve", use_container_width=True):
        st.session_state.selected_sample_key = "valve_abnormal"
with btn_c5:
    if st.button("⚠️ Rail Friction", use_container_width=True):
        st.session_state.selected_sample_key = "slider_abnormal"

# Dropdown or Upload
col_sample_select, col_upload = st.columns([1.3, 1])

preset_keys = list(sample_presets.keys())
preset_names = [sample_presets[k]["name"] for k in preset_keys]

default_idx = preset_keys.index(st.session_state.selected_sample_key) if st.session_state.selected_sample_key in preset_keys else 0

with col_sample_select:
    chosen_preset_name = st.selectbox(
        "Or choose any machine condition from list:",
        options=preset_names,
        index=default_idx,
        help="Select any normal or faulty equipment audio file from the MIMII benchmark dataset."
    )
    # Sync dropdown selection with state
    for k, v in sample_presets.items():
        if v["name"] == chosen_preset_name:
            st.session_state.selected_sample_key = k
            break

with col_upload:
    uploaded_file = st.file_uploader(
        "Or upload custom audio (.wav, 16 kHz):",
        type=["wav"],
        help="Upload any 16,000 Hz industrial audio WAV file."
    )

# Determine Active File Path
active_file_path = None
active_sample_desc = ""

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        active_file_path = tmp.name
        active_sample_desc = f"Custom Upload: `{uploaded_file.name}`"
else:
    current_preset = sample_presets[st.session_state.selected_sample_key]
    active_file_path = current_preset["path"]
    active_sample_desc = current_preset["desc"]

if active_file_path is None or not os.path.exists(active_file_path):
    st.warning("⚠️ Could not locate audio file. Please pick another sample or upload a file.")
    st.stop()

# Audio Preprocessing
try:
    sr, raw_audio = load_wav_file(active_file_path)
    mono_audio = mix_channels(raw_audio, mode='mean')
    norm_audio = normalize_amplitude(mono_audio, target_peak=1.0)
    spectrograms = process_file(active_file_path)
    spec = spectrograms[len(spectrograms) // 2]
except Exception as err:
    st.error(f"Error processing audio: {err}")
    st.stop()

num_channels = 1 if raw_audio.ndim == 1 else raw_audio.shape[1]
duration_sec = len(raw_audio) / float(sr)

# Audio Player & Signal Specs Bar
st.markdown("---")
st.markdown("### 🔊 Step 2: Listen & Inspect Acoustic Signal")

if active_sample_desc:
    st.caption(f"ℹ️ **Sample Description:** {active_sample_desc}")

player_c, stat_c1, stat_c2, stat_c3 = st.columns([1.8, 1, 1, 1])

with player_c:
    st.audio(active_file_path, format="audio/wav")

with stat_c1:
    st.metric("Sampling Rate", f"{sr:,} Hz", help="16,000 Hz is standard for industrial acoustic anomaly detection.")
with stat_c2:
    st.metric("Duration", f"{duration_sec:.1f} sec", help="Audio clip length analyzed by the model.")
with stat_c3:
    st.metric("Signal Type", "8-ch Array" if num_channels > 1 else "Mono Channel", help="Spatial microphone array or single mic channel.")

# Step 3: Run Model Inference
tensor_spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    logits = active_model(tensor_spec)
    probs = F.softmax(logits, dim=1).numpy()[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

pred_class = CLASSES[pred_idx]
machine_type, condition = pred_class.split('_')
is_normal = (condition.lower() == 'normal')

st.markdown("---")
st.markdown("### 🎯 Step 3: Machine Health Diagnosis")

# Equipment Icon Mapping
machine_icons = {"Fan": "🌀", "Pump": "🚰", "Valve": "🎚️", "SlideRail": "🛤️", "Slider": "🛤️"}
m_icon = machine_icons.get(machine_type, "⚙️")

# Recommended Actions Database
actions = {
    "Fan": {
        "normal": "Machine is running smoothly. Continue routine schedule; inspect again during normal maintenance cycle.",
        "anomaly": "Urgent Inspection: Check motor bearings for wear, test blade balance, and verify mounting bolt tightness."
    },
    "Pump": {
        "normal": "Flow rate and hydraulic pressure are optimal. Acoustic signature shows steady lubrication.",
        "anomaly": "Action Required: Check for impeller cavitation, inspect suction line filters, and evaluate seal integrity."
    },
    "Valve": {
        "normal": "Valve seat seals tightly without backpressure leaks or turbulence.",
        "anomaly": "Action Required: Inspect sealing gasket for fluid/gas leaks, check actuator pressure, and clean seat surface."
    },
    "SlideRail": {
        "normal": "Guide carriage moves freely along rails without excessive friction.",
        "anomaly": "Action Required: Lubricate track bearings, clean guide rails of debris, and verify rail alignment."
    }
}

machine_key = "SlideRail" if "slide" in machine_type.lower() else machine_type
rec_text = actions.get(machine_key, {}).get("normal" if is_normal else "anomaly", "Review machine parameters.")

# Render Diagnosis Card
if is_normal:
    st.markdown(f"""
    <div class="status-box-normal">
        <div style="font-size:1.55rem; font-weight:800; display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
            <span>✅</span> HEALTHY — NORMAL OPERATING CONDITION
        </div>
        <div style="font-size:1.05rem; margin-bottom:0.5rem;">
            Equipment Detected: <b>{m_icon} {machine_type}</b> &nbsp;|&nbsp;
            Diagnosis: <span class="pill-label pill-normal">Normal</span> &nbsp;|&nbsp;
            Confidence: <b>{confidence*100:.1f}%</b>
        </div>
        <div style="font-size:0.95rem; color:#166534;">
            <b>Acoustic Assessment:</b> Sound matches clean baseline patterns. Harmonic frequencies are stable without anomalous friction or rattle.
        </div>
        <div class="action-box" style="border-left-color: #22c55e; background: #ffffff;">
            🛠️ <b>Recommended Action:</b> {rec_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="status-box-anomaly">
        <div style="font-size:1.55rem; font-weight:800; display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
            <span>⚠️</span> ALERT — ACOUSTIC FAULT / ANOMALY DETECTED
        </div>
        <div style="font-size:1.05rem; margin-bottom:0.5rem;">
            Equipment Detected: <b>{m_icon} {machine_type}</b> &nbsp;|&nbsp;
            Diagnosis: <span class="pill-label pill-anomaly">Defect / Anomaly</span> &nbsp;|&nbsp;
            Confidence: <b>{confidence*100:.1f}%</b>
        </div>
        <div style="font-size:0.95rem; color:#991b1b;">
            <b>Acoustic Assessment:</b> Sound deviates significantly from normal operating standards. Irregular frequency pulses or friction spikes detected.
        </div>
        <div class="action-box" style="border-left-color: #ef4444; background: #ffffff;">
            🛠️ <b>Recommended Action:</b> {rec_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Deep-Dive Analytics Tabs
tab_prob, tab_spectrogram, tab_explain = st.tabs([
    "📊 Probability Breakdown",
    "📈 Sound Waveform & Spectrogram (Acoustic Fingerprint)",
    "🔍 AI Explainability (Grad-CAM Heatmap)"
])

# TAB 1: Probability Breakdown
with tab_prob:
    st.subheader("Model Confidence Distribution")
    st.caption("How strongly the neural network voted for each possible machine operating state:")
    
    df_probs = pd.DataFrame({
        'Class': CLASSES,
        'Probability (%)': [p * 100 for p in probs]
    }).sort_values(by='Probability (%)', ascending=True)
    
    fig_prob, ax_prob = plt.subplots(figsize=(10, 4.2))
    colors = [
        '#22c55e' if (c == pred_class and is_normal) 
        else ('#ef4444' if (c == pred_class and not is_normal) else '#cbd5e1') 
        for c in df_probs['Class']
    ]
    bars = ax_prob.barh(df_probs['Class'], df_probs['Probability (%)'], color=colors, height=0.6)
    
    for bar in bars:
        w = bar.get_width()
        if w > 2:
            ax_prob.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=9, fontweight='600')
            
    ax_prob.set_xlabel("Confidence Probability (%)", fontsize=10, fontweight='600')
    ax_prob.set_xlim(0, 105)
    ax_prob.grid(True, linestyle='--', alpha=0.3, axis='x')
    ax_prob.spines['top'].set_visible(False)
    ax_prob.spines['right'].set_visible(False)
    st.pyplot(fig_prob)
    
    st.info("""
    💡 **How to interpret this chart:**
    - The longest bar shows the AI's primary diagnosis.
    - Green indicates a healthy condition, while Red signals an anomaly.
    - A confidence score above **70%** indicates high model certainty.
    """)

# TAB 2: Audio Signals & Spectrogram
with tab_spectrogram:
    st.subheader("Visualizing the Machine's Sound")
    st.caption("Sound converted into visual representations to reveal patterns the human ear might miss.")
    
    col_w, col_s = st.columns(2)
    
    with col_w:
        st.markdown("##### 1. Time-Domain Waveform")
        st.caption("Shows sound volume changes over time. Sudden spikes often indicate impacts, friction, or loose parts.")
        fig_w, ax_w = plt.subplots(figsize=(6, 3.8))
        time_ax = np.linspace(0, duration_sec, len(norm_audio))
        ax_w.plot(time_ax, norm_audio, color='#1e3a8a', alpha=0.85, linewidth=0.7)
        ax_w.set_xlabel("Time (seconds)", fontsize=9)
        ax_w.set_ylabel("Normalized Amplitude", fontsize=9)
        ax_w.grid(True, linestyle='--', alpha=0.3)
        ax_w.spines['top'].set_visible(False)
        ax_w.spines['right'].set_visible(False)
        st.pyplot(fig_w)
        
    with col_s:
        st.markdown("##### 2. 64-Bin Log-Mel Spectrogram")
        st.caption("The acoustic fingerprint: Horizontal = Time, Vertical = Pitch (Frequency), Color = Loudness.")
        fig_s, ax_s = plt.subplots(figsize=(6, 3.8))
        im_s = ax_s.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        ax_s.set_xlabel("Time Frames", fontsize=9)
        ax_s.set_ylabel("Mel Frequency Bins (Low → High)", fontsize=9)
        fig_s.colorbar(im_s, ax=ax_s, format='%+2.0f dB')
        st.pyplot(fig_s)

# TAB 3: Grad-CAM Explainability
with tab_explain:
    st.subheader("Grad-CAM: Seeing What the AI 'Heard'")
    st.caption("Understand why the AI made its diagnosis by inspecting the exact acoustic frequencies that triggered the neural network.")
    
    if "ResNet18" in model_name_str:
        try:
            tensor_spec_grad = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            tensor_spec_grad.requires_grad = True
            
            grad_cam = GradCAM(active_model, active_model.resnet.layer4[-1])
            heatmap = grad_cam.generate_heatmap(tensor_spec_grad, class_idx=pred_idx)
            
            fig_g, axes_g = plt.subplots(1, 3, figsize=(14, 4.2))
            
            # Mel Spectrogram
            axes_g[0].imshow(spec, aspect='auto', origin='lower', cmap='magma')
            axes_g[0].set_title("Input Sound Spectrogram", fontsize=10, fontweight='bold')
            axes_g[0].set_xlabel("Time Frames")
            axes_g[0].set_ylabel("Frequency Bins")
            
            # Heatmap
            axes_g[1].imshow(heatmap, aspect='auto', origin='lower', cmap='jet')
            axes_g[1].set_title(f"AI Focus Map ({pred_class})", fontsize=10, fontweight='bold')
            axes_g[1].set_xlabel("Time Frames")
            
            # Overlay
            axes_g[2].imshow(spec, aspect='auto', origin='lower', cmap='gray')
            axes_g[2].imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
            axes_g[2].set_title("Suspicious Sound Regions Overlaid", fontsize=10, fontweight='bold')
            axes_g[2].set_xlabel("Time Frames")
            
            plt.tight_layout()
            st.pyplot(fig_g)
            
            st.markdown("""
            <div class="card-box" style="background:#f8fafc; border-left:4px solid #f59e0b;">
                <h5 style="margin-top:0; color:#b45309;">🔎 How to Read this Heatmap:</h5>
                <ul style="margin-bottom:0; font-size:0.92rem; color:#334155; line-height:1.6;">
                    <li><b>🔴 Red & Yellow Hotspots:</b> Highlight the specific pitch and exact moments in time where abnormal acoustic energy was detected.</li>
                    <li><b>🔵 Cool Blue Regions:</b> Background ambient factory hum that the deep learning model determined was harmless and ignored.</li>
                    <li><b>Industrial Value:</b> Instead of an unexplainable 'black-box' alert, maintenance teams can pinpoint whether the defect is a continuous high-pitch whine (e.g. bearing failure) or intermittent low rumbles.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e_cam:
            st.warning(f"Grad-CAM could not be computed: {e_cam}")
    else:
        st.info("ℹ️ Grad-CAM explainability heatmaps are enabled for the **ResNet18 Transfer Learning** model. Select ResNet18 in the sidebar to view feature activation maps.")

# Clean up temporary uploaded file if needed
if uploaded_file is not None and active_file_path and os.path.exists(active_file_path):
    try:
        os.unlink(active_file_path)
    except Exception:
        pass
