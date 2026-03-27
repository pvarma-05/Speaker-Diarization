"""Audio preprocessing utilities."""

import librosa
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


def load_audio(file_path: str, target_sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio file and convert to target sample rate.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default: 16000 Hz)
        mono: Convert to mono if True

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
        return audio, sr
    except Exception as e:
        raise ValueError(f"Failed to load audio file {file_path}: {e}")
