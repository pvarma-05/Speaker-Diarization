"""
Open-set speaker identification using ECAPA-TDNN pretrained embeddings (SpeechBrain).
Compares input embeddings with known speaker registry using cosine similarity.
"""

import numpy as np
import torch
import torchaudio
import os
import json
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_SAVEDIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pretrained_models", "ecapa-tdnn"
)


def load_speaker_encoder():
    """Load pretrained ECAPA-TDNN speaker encoder from SpeechBrain."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = EncoderClassifier.from_hparams(
            source=ECAPA_SOURCE,
            savedir=ECAPA_SAVEDIR,
            run_opts={"device": device}
        )
        return encoder
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ECAPA-TDNN encoder: {e}\n"
            "Install speechbrain: pip install speechbrain"
        )


def extract_speaker_embedding(audio_file: str, encoder=None) -> np.ndarray:
    """Extract 192-dim ECAPA-TDNN embedding from an audio file."""
    if encoder is None:
        encoder = load_speaker_encoder()

    waveform, sr = torchaudio.load(audio_file)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    embedding = encoder.encode_batch(waveform)
    return embedding.squeeze().cpu().numpy()


def extract_embedding_from_segment(audio_array: np.ndarray, sr: int,
                                   encoder=None) -> np.ndarray:
    """Extract 192-dim ECAPA-TDNN embedding from a numpy audio segment."""
    if encoder is None:
        encoder = load_speaker_encoder()

    waveform = torch.tensor(audio_array, dtype=torch.float32)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    embedding = encoder.encode_batch(waveform)
    return embedding.squeeze().cpu().numpy()


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings (-1 to 1)."""
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def load_speaker_registry(registry_path: str = "data/speaker_registry.json") -> Dict:
    """Load known speaker embeddings from registry JSON file."""
    if not os.path.exists(registry_path):
        return {}

    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)

        for speaker_name, data in registry.items():
            if 'embedding' in data:
                data['embedding'] = np.array(data['embedding'])

        return registry
    except Exception as e:
        print(f"Warning: Failed to load speaker registry: {e}")
        return {}


def save_speaker_to_registry(speaker_name: str, embedding: np.ndarray,
                             registry_path: str = "data/speaker_registry.json",
                             metadata: Dict = None):
    """Save a speaker embedding to the registry."""
    registry = load_speaker_registry(registry_path)

    # registry[speaker_name] = {
    #     'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
    #     'metadata': metadata or {}
    # }
    if speaker_name in registry:
        old_emb = np.array(registry[speaker_name]['embedding'])
        new_emb = embedding
    
        final_emb = (old_emb + new_emb) / 2
    else:
        final_emb = embedding
    
    registry[speaker_name] = {
        'embedding': final_emb.tolist(),
        'metadata': metadata or {}
    }

    os.makedirs(os.path.dirname(registry_path) or '.', exist_ok=True)

    serializable = {}
    for name, data in registry.items():
        emb = data['embedding']
        serializable[name] = {
            'embedding': emb.tolist() if isinstance(emb, np.ndarray) else emb,
            'metadata': data.get('metadata', {})
        }

    with open(registry_path, 'w') as f:
        json.dump(serializable, f, indent=2)


def identify_speaker(embedding: np.ndarray, registry: Dict = None,
                     threshold: float = 0.75,
                     registry_path: str = "data/speaker_registry.json") -> Tuple[str, float]:
    """
    Identify a speaker by comparing embedding against the registry.
    Returns ("UNKNOWN", best_score) if no match exceeds the threshold.
    """
    if registry is None:
        registry = load_speaker_registry(registry_path)

    if not registry:
        return ("UNKNOWN", 0.0)

    best_match = None
    best_similarity = -1.0

    for speaker_name, data in registry.items():
        if 'embedding' in data:
            similarity = cosine_similarity(embedding, data['embedding'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_name

    if best_similarity >= threshold:
        return (best_match, best_similarity)
    return ("UNKNOWN", best_similarity)


def resolve_speaker_display(
    diarization_label: str,
    embedding: Optional[np.ndarray],
    registry: Dict,
    threshold: float,
    dia_to_guest: Dict[str, str],
    guest_counter: List[int],
) -> Tuple[str, str]:
    """
    Map diarization label + embedding to display speaker and AUTHORIZED/UNAUTHORIZED.

    If registry match: speaker = enrolled name, type AUTHORIZED.
    Else: speaker = Speaker_N (stable per diarization_label), type UNAUTHORIZED.

    guest_counter is a single-element list [int] for mutable counter.
    """
    if embedding is not None and registry:
        name, score = identify_speaker(embedding, registry, threshold)
        # DEBUG STATEMENT
        # print(f"Similarity: {score}, matched: {name}")
        if name != "UNKNOWN":
            return name, "AUTHORIZED"

    if diarization_label not in dia_to_guest:
        guest_counter[0] += 1
        dia_to_guest[diarization_label] = f"Speaker_{guest_counter[0]}"

    return dia_to_guest[diarization_label], "UNAUTHORIZED"


def identify_speakers_in_segments(aligned_segments: List[Dict], audio_file: str,
                                  registry_path: str = "data/speaker_registry.json",
                                  threshold: float = 0.75) -> List[Dict]:
    """Legacy: enrich segments with identified_speaker (kept for scripts)."""
    import librosa

    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    encoder = load_speaker_encoder()
    registry = load_speaker_registry(registry_path)

    identified_segments = []

    for seg in aligned_segments:
        start_time = seg['start']
        end_time = seg['end']

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment_audio = audio[start_sample:end_sample]

        new_seg = seg.copy()

        if len(segment_audio) < sr * 0.5:
            new_seg['identified_speaker'] = 'UNKNOWN'
            new_seg['similarity_score'] = 0.0
            identified_segments.append(new_seg)
            continue

        try:
            embedding = extract_embedding_from_segment(segment_audio, sr, encoder)
            speaker_name, similarity = identify_speaker(
                embedding, registry, threshold, registry_path
            )
            new_seg['identified_speaker'] = speaker_name
            new_seg['similarity_score'] = similarity
        except Exception as e:
            print(f"  Warning: Speaker ID failed for [{start_time:.2f}-{end_time:.2f}]: {e}")
            new_seg['identified_speaker'] = 'UNKNOWN'
            new_seg['similarity_score'] = 0.0

        identified_segments.append(new_seg)

    return identified_segments


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python speaker_id.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    print(f"Extracting ECAPA-TDNN embedding from {audio_file}...")

    embedding = extract_speaker_embedding(audio_file)
    print(f"Embedding shape: {embedding.shape}")

    speaker_name, similarity = identify_speaker(embedding)
    print(f"Identified as: {speaker_name} (similarity: {similarity:.3f})")
