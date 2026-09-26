import numpy as np
from scipy import signal
from src.utils import load_wav_file

def mix_channels(audio: np.ndarray, mode: str = 'mean') -> np.ndarray:
    """
    Convert multi-channel audio (e.g. 8-channel MIMII array) to 1 mono channel.
    Modes:
      - 'mean': average across all channels
      - 'channel_0': select channel index 0
    """
    if audio.ndim == 1:
        return audio
    if mode == 'mean':
        return np.mean(audio, axis=1)
    elif mode == 'channel_0':
        return audio[:, 0]
    else:
        raise ValueError(f"Unsupported channel mixing mode: '{mode}'")

def normalize_amplitude(audio: np.ndarray, target_peak: float = 1.0) -> np.ndarray:
    """
    Peak amplitude normalization to [-target_peak, target_peak].
    """
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        return (audio / peak) * target_peak
    return audio

def segment_audio(
    audio: np.ndarray,
    window_seconds: float = 2.0,
    hop_seconds: float = 1.0,
    sample_rate: int = 16000
) -> list:
    """
    Segment audio into fixed-duration overlapping or non-overlapping windows.
    Returns list of 1D numpy float arrays.
    """
    win_len = int(window_seconds * sample_rate)
    hop_len = int(hop_seconds * sample_rate)
    
    if len(audio) < win_len:
        # Pad with zeros if shorter than window
        padded = np.zeros(win_len, dtype=audio.dtype)
        padded[:len(audio)] = audio
        return [padded]
        
    segments = []
    start = 0
    while start + win_len <= len(audio):
        segments.append(audio[start : start + win_len])
        start += hop_len
        
    return segments

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def _get_mel_filterbank(sr: int, n_fft: int, n_mels: int = 64, fmin: float = 0.0, fmax: float = None):
    """Generate triangular Mel-scale filterbank matrix."""
    if fmax is None:
        fmax = sr / 2.0
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    bank = np.zeros((n_mels, int(n_fft // 2 + 1)), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                bank[m - 1, k] = (k - bin_points[m - 1]) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                bank[m - 1, k] = (bin_points[m + 1] - k) / (f_m_plus - f_m)
    return bank

def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64
) -> np.ndarray:
    """
    Compute dB-scaled Mel-Spectrogram from 1D mono audio array.
    Output shape: (n_mels, time_frames)
    """
    _, _, Sxx = signal.spectrogram(
        audio,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        mode='magnitude'
    )
    filterbank = _get_mel_filterbank(sample_rate, n_fft, n_mels=n_mels)
    mel_spec = np.dot(filterbank, Sxx)
    mel_db = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    return mel_db

def process_file(
    filepath: str,
    mix_mode: str = 'mean',
    target_peak: float = 1.0,
    window_seconds: float = 2.0,
    hop_seconds: float = 1.0,
    n_mels: int = 64
) -> list:
    """
    Full Preprocessing Pipeline:
    .wav → Load → Mix Channels → Normalize → Segment → Mel-Spectrogram
    Returns list of 2D Mel-Spectrogram arrays (one per segment).
    """
    sr, raw_audio = load_wav_file(filepath)
    mono_audio = mix_channels(raw_audio, mode=mix_mode)
    norm_audio = normalize_amplitude(mono_audio, target_peak=target_peak)
    segments = segment_audio(
        norm_audio,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        sample_rate=sr
    )
    spectrograms = [
        compute_mel_spectrogram(seg, sample_rate=sr, n_mels=n_mels)
        for seg in segments
    ]
    return spectrograms
