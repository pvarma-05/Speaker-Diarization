"""
Open-set speaker identification using pretrained speaker embeddings.
Compares input embeddings with known speaker registry using cosine similarity.
"""

import numpy as np
from typing import List, Dict, Tuple
import os
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    print("Warning: resemblyzer not available. Using placeholder implementation.")


def load_speaker_encoder():
    """
    Load pretrained speaker encoder model.
    
    Returns:
        VoiceEncoder model or None if not available
    """
    if not RESEMBLYZER_AVAILABLE:
        return None
    
    try:
        encoder = VoiceEncoder()
        return encoder
    except Exception as e:
        print(f"Warning: Failed to load speaker encoder: {str(e)}")
        return None


def extract_speaker_embedding(audio_file: str, encoder=None) -> np.ndarray:
    """
    Extract speaker embedding from audio file.
    
    Args:
        audio_file: Path to audio file
        encoder: Optional pre-loaded encoder
        
    Returns:
        Speaker embedding vector
    """
    if encoder is None:
        encoder = load_speaker_encoder()
    
    if encoder is None:
        # Return dummy embedding if encoder not available
        return np.random.randn(256)
    
    try:
        wav = preprocess_wav(audio_file)
        embedding = encoder.embed_utterance(wav)
        return embedding
    except Exception as e:
        raise RuntimeError(f"Failed to extract speaker embedding: {str(e)}")


def extract_embedding_from_segment(audio_array: np.ndarray, sr: int, 
                                   encoder=None) -> np.ndarray:
    """
    Extract speaker embedding from audio array segment.
    
    Args:
        audio_array: Audio array
        sr: Sample rate
        encoder: Optional pre-loaded encoder
        
    Returns:
        Speaker embedding vector
    """
    if encoder is None:
        encoder = load_speaker_encoder()
    
    if encoder is None:
        return np.random.randn(256)
    
    try:
        # Resemblyzer expects 16kHz mono audio
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        
        embedding = encoder.embed_utterance(audio_array)
        return embedding
    except Exception as e:
        raise RuntimeError(f"Failed to extract embedding from segment: {str(e)}")


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        
    Returns:
        Cosine similarity score (-1 to 1)
    """
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def load_speaker_registry(registry_path: str = "data/speaker_registry.json") -> Dict:
    """
    Load known speaker embeddings registry from JSON file.
    
    Expected format:
    {
        "speaker_name": {
            "embedding": [list of floats],
            "metadata": {...}
        }
    }
    
    Args:
        registry_path: Path to registry JSON file
        
    Returns:
        Dictionary mapping speaker names to embeddings
    """
    if not os.path.exists(registry_path):
        # Return empty registry if file doesn't exist
        return {}
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Convert embeddings back to numpy arrays
        for speaker_name, data in registry.items():
            if 'embedding' in data:
                data['embedding'] = np.array(data['embedding'])
        
        return registry
    except Exception as e:
        print(f"Warning: Failed to load speaker registry: {str(e)}")
        return {}


def save_speaker_to_registry(speaker_name: str, embedding: np.ndarray,
                             registry_path: str = "data/speaker_registry.json",
                             metadata: Dict = None):
    """
    Save speaker embedding to registry.
    
    Args:
        speaker_name: Name of speaker
        embedding: Speaker embedding vector
        registry_path: Path to registry JSON file
        metadata: Optional metadata dictionary
    """
    registry = load_speaker_registry(registry_path)
    
    # Convert numpy array to list for JSON serialization
    embedding_list = embedding.tolist()
    
    registry[speaker_name] = {
        'embedding': embedding_list,
        'metadata': metadata or {}
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    
    # Convert any numpy arrays back to lists for JSON serialization
    serializable_registry = {}
    for name, data in registry.items():
        serializable_registry[name] = {
            'embedding': data['embedding'].tolist() if isinstance(data['embedding'], np.ndarray) else data['embedding'],
            'metadata': data.get('metadata', {})
        }
    
    with open(registry_path, 'w') as f:
        json.dump(serializable_registry, f, indent=2)


def identify_speaker(embedding: np.ndarray, registry: Dict = None,
                    threshold: float = 0.75, registry_path: str = "data/speaker_registry.json") -> Tuple[str, float]:
    """
    Identify speaker by comparing embedding with known speakers in registry.
    
    Args:
        embedding: Speaker embedding to identify
        registry: Optional pre-loaded registry (if None, will load from file)
        threshold: Similarity threshold for identification (default: 0.7)
        registry_path: Path to registry file
        
    Returns:
        Tuple of (speaker_name, similarity_score)
        Returns ("UNKNOWN", 0.0) if no match above threshold
    """
    if registry is None:
        registry = load_speaker_registry(registry_path)
    
    if not registry:
        return ("UNKNOWN", 0.0)
    
    best_match = None
    best_similarity = -1.0
    
    for speaker_name, data in registry.items():
        if 'embedding' in data:
            known_embedding = data['embedding']
            similarity = cosine_similarity(embedding, known_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_name
    
    if best_similarity >= threshold:
        return (best_match, best_similarity)
    else:
        return ("UNKNOWN", best_similarity)


def identify_speakers_in_segments(aligned_segments: List[Dict], audio_file: str,
                                  registry_path: str = "data/speaker_registry.json",
                                  threshold: float = 0.75) -> List[Dict]:
    """
    Identify speakers for each aligned segment.
    
    Args:
        aligned_segments: List of aligned segments with speaker labels
        audio_file: Path to original audio file
        registry_path: Path to speaker registry
        threshold: Similarity threshold
        
    Returns:
        List of segments with updated speaker names (or UNKNOWN)
    """
    import librosa
    
    # Load audio
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    
    # Load encoder and registry
    encoder = load_speaker_encoder()
    registry = load_speaker_registry(registry_path)
    
    identified_segments = []
    
    for seg in aligned_segments:
        start_time = seg['start']
        end_time = seg['end']
        diarization_label = seg['speaker_label']
        
        # Extract audio segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment_audio = audio[start_sample:end_sample]
        
        if len(segment_audio) < sr * 0.5:  # Skip segments shorter than 0.5 seconds
            identified_segments.append(seg.copy())
            continue
        
        # Extract embedding
        try:
            embedding = extract_embedding_from_segment(segment_audio, sr, encoder)
            
            # Identify speaker
            speaker_name, similarity = identify_speaker(embedding, registry, threshold, registry_path)
            
            # Update segment
            new_seg = seg.copy()
            new_seg['identified_speaker'] = speaker_name
            new_seg['similarity_score'] = similarity
            identified_segments.append(new_seg)
        except Exception as e:
            print(f"Warning: Failed to identify speaker for segment [{start_time:.2f}-{end_time:.2f}]: {str(e)}")
            new_seg = seg.copy()
            new_seg['identified_speaker'] = "UNKNOWN"
            new_seg['similarity_score'] = 0.0
            identified_segments.append(new_seg)
    
    return identified_segments


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python speaker_id.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    print(f"Extracting speaker embedding from {audio_file}...")
    
    embedding = extract_speaker_embedding(audio_file)
    print(f"Embedding shape: {embedding.shape}")
    
    speaker_name, similarity = identify_speaker(embedding)
    print(f"Identified as: {speaker_name} (similarity: {similarity:.3f})")
