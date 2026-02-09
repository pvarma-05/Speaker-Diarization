"""
Automatic Speech Recognition using WhisperX pretrained model.
Generates text transcripts with word-level timestamps.
"""

import whisperx
import torch
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


def load_whisperx_model(model_name: str = "base", device: str = None, 
                       compute_type: str = "int8") -> Dict:
    """
    Load WhisperX model and alignment model.
    
    Args:
        model_name: Whisper model size (tiny, base, small, medium, large)
        device: Device to use (cuda/cpu). Auto-detected if None
        compute_type: Compute type for quantization
        
    Returns:
        Dictionary with 'model' and 'align_model' keys
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load WhisperX model
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        
        # Load alignment model (for word-level timestamps)
        align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
        
        return {
            'model': model,
            'align_model': align_model,
            'metadata': metadata,
            'device': device
        }
    except Exception as e:
        raise RuntimeError(f"Failed to load WhisperX model: {str(e)}")


def transcribe_audio(audio_file: str, models: Dict = None, 
                     model_name: str = "base", batch_size: int = 16) -> List[Dict]:
    """
    Transcribe audio file with word-level timestamps.
    
    Args:
        audio_file: Path to audio file
        models: Optional pre-loaded models dict
        model_name: Whisper model size (if models not provided)
        batch_size: Batch size for processing
        
    Returns:
        List of dictionaries with keys: 'start', 'end', 'text'
    """
    if models is None:
        models = load_whisperx_model(model_name)
    
    try:
        # Load audio
        audio = whisperx.load_audio(audio_file)
        
        # Transcribe
        result = models['model'].transcribe(audio, batch_size=batch_size)
        
        # Align for word-level timestamps
        result = whisperx.align(result["segments"], models['align_model'], 
                               models['metadata'], audio, models['device'], 
                               return_char_alignments=False)
        
        # Extract segments with timestamps
        segments = []
        for segment in result["segments"]:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        
        return segments
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")


def format_transcription_output(segments: List[Dict]) -> str:
    """
    Format transcription segments as human-readable string.
    
    Args:
        segments: List of transcription segments
        
    Returns:
        Formatted string
    """
    output = []
    for seg in segments:
        output.append(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}")
    return "\n".join(output)


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    print(f"Transcribing {audio_file}...")
    
    segments = transcribe_audio(audio_file)
    print(f"\nFound {len(segments)} transcription segments:")
    print(format_transcription_output(segments))
