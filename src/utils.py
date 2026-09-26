import os
import numpy as np
from scipy.io import wavfile

# Milestone-specified class labels
CLASSES = [
    'Fan_Normal',
    'Fan_Anomaly',
    'Pump_Normal',
    'Pump_Anomaly',
    'SlideRail_Normal',
    'SlideRail_Anomaly',
    'Valve_Normal',
    'Valve_Anomaly'
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

# Legacy/Folder label to Standardized Label mapping
LABEL_MAP = {
    'Fan_Normal': 'Fan_Normal',
    'Fan_Abnormal': 'Fan_Anomaly',
    'Fan_Anomaly': 'Fan_Anomaly',
    'Pump_Normal': 'Pump_Normal',
    'Pump_Abnormal': 'Pump_Anomaly',
    'Pump_Anomaly': 'Pump_Anomaly',
    'Slider_Normal': 'SlideRail_Normal',
    'Slider_Abnormal': 'SlideRail_Anomaly',
    'SlideRail_Normal': 'SlideRail_Normal',
    'SlideRail_Anomaly': 'SlideRail_Anomaly',
    'Valve_Normal': 'Valve_Normal',
    'Valve_Abnormal': 'Valve_Anomaly',
    'Valve_Anomaly': 'Valve_Anomaly'
}

def standardize_label(label: str) -> str:
    """Map legacy label strings to standard milestone label format."""
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    raise ValueError(f"Unknown label string: '{label}'")

def resolve_filepath(path: str, base_dir: str = None) -> str:
    """
    Resolve Google Colab/Drive hardcoded paths, relative paths,
    or different workspace root directories to valid local workspace paths.
    """
    if not path:
        return path
        
    norm_path = path.replace('\\', '/')
    
    # Extract relative path starting at dataset/MIMII
    if 'dataset/MIMII/' in norm_path:
        rel_part = norm_path.split('dataset/MIMII/')[1]
        relative_target = os.path.join('dataset', 'MIMII', rel_part.replace('/', os.sep))
    else:
        relative_target = norm_path.replace('/', os.sep)

    candidate_bases = []
    if base_dir:
        candidate_bases.append(base_dir)
    candidate_bases.append(os.getcwd())
    candidate_bases.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Colab candidate bases
    if os.path.exists('/content'):
        candidate_bases.append('/content')
        for item in os.listdir('/content'):
            item_p = os.path.join('/content', item)
            if os.path.isdir(item_p):
                candidate_bases.append(item_p)

    for b in candidate_bases:
        cand = os.path.join(b, relative_target)
        if os.path.exists(cand):
            return cand
            
    # Fallback to direct path or base_dir path
    if base_dir and not os.path.isabs(relative_target):
        return os.path.join(base_dir, relative_target)
        
    return relative_target


def load_wav_file(filepath: str):
    """
    Robust audio loading using scipy.io.wavfile.
    Returns: (sample_rate, float32_audio_array)
    Handles 8-channel WAVE_FORMAT_EXTENSIBLE files seamlessly.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found at path: '{filepath}'")
        
    sr, data = wavfile.read(filepath)
    # Convert integer types to float32 normalized to [-1.0, 1.0]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    return sr, data
