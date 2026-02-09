"""
Audio preprocessing utilities including format conversion and Voice Activity Detection (VAD).
"""

import librosa
import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


def load_audio(file_path: str, target_sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio file and convert to target sample rate and mono if needed.
    
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
        raise ValueError(f"Failed to load audio file {file_path}: {str(e)}")


def apply_vad_simple(audio: np.ndarray, sr: int, frame_length: int = 2048, 
                     hop_length: int = 512, energy_threshold: float = 0.01) -> List[Tuple[float, float]]:
    """
    Simple energy-based VAD to detect speech segments.
    
    Args:
        audio: Audio array
        sr: Sample rate
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation
        energy_threshold: Energy threshold for speech detection
        
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    # Calculate frame energy
    frame_energies = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        energy = np.mean(frame ** 2)
        frame_energies.append(energy)
    
    # Convert to numpy array
    frame_energies = np.array(frame_energies)
    
    # Threshold to get speech frames
    speech_frames = frame_energies > energy_threshold
    
    # Find continuous speech segments
    segments = []
    in_speech = False
    start_idx = 0
    
    for i, is_speech in enumerate(speech_frames):
        if is_speech and not in_speech:
            # Start of speech segment
            start_idx = i
            in_speech = True
        elif not is_speech and in_speech:
            # End of speech segment
            end_idx = i
            start_time = (start_idx * hop_length) / sr
            end_time = (end_idx * hop_length) / sr
            segments.append((start_time, end_time))
            in_speech = False
    
    # Handle case where audio ends during speech
    if in_speech:
        end_idx = len(speech_frames)
        start_time = (start_idx * hop_length) / sr
        end_time = len(audio) / sr
        segments.append((start_time, end_time))
    
    return segments


def remove_silence(audio: np.ndarray, sr: int, frame_length: int = 2048,
                   hop_length: int = 512, energy_threshold: float = 0.01) -> np.ndarray:
    """
    Remove silence from audio using simple energy-based VAD.
    
    Args:
        audio: Audio array
        sr: Sample rate
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation
        energy_threshold: Energy threshold for speech detection
        
    Returns:
        Audio array with silence removed
    """
    segments = apply_vad_simple(audio, sr, frame_length, hop_length, energy_threshold)
    
    if not segments:
        return np.array([])
    
    # Concatenate all speech segments
    speech_samples = []
    for start_time, end_time in segments:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        speech_samples.append(audio[start_sample:end_sample])
    
    if speech_samples:
        return np.concatenate(speech_samples)
    return np.array([])
