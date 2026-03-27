"""
Speaker diarization using pretrained pyannote.audio pipeline.
Outputs speaker segments with start/end timestamps and speaker labels.
"""

from pyannote.audio import Pipeline
import torch
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


def load_diarization_pipeline(use_auth_token: str = None) -> Pipeline:
    """
    Load pretrained pyannote.audio diarization pipeline.
    
    Args:
        use_auth_token: HuggingFace token for model access
        
    Returns:
        pyannote.audio Pipeline object
    """
    try:
        if use_auth_token:
            import os
            os.environ["HF_TOKEN"] = use_auth_token

        # Newer pyannote uses 'token', older uses 'use_auth_token'
        import inspect
        sig = inspect.signature(Pipeline.from_pretrained)
        if 'token' in sig.parameters:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=use_auth_token
            )
        else:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=use_auth_token
            )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
        
        return pipeline
    except Exception as e:
        raise RuntimeError(f"Failed to load diarization pipeline: {str(e)}\n"
                          "Note: You may need to accept model terms at "
                          "https://huggingface.co/pyannote/speaker-diarization-3.1")


def diarize_audio(audio_file: str, pipeline: Pipeline = None, 
                  use_auth_token: str = None) -> List[Dict]:
    """
    Perform speaker diarization on audio file.
    
    Args:
        audio_file: Path to audio file
        pipeline: Optional pre-loaded pipeline (if None, will load one)
        use_auth_token: HuggingFace token if needed
        
    Returns:
        List of dictionaries with keys: 'start', 'end', 'speaker_label'
    """
    if pipeline is None:
        pipeline = load_diarization_pipeline(use_auth_token)
    
    try:
        # Pre-load audio with torchaudio to avoid torchcodec issues on Windows
        import torchaudio
        waveform, sample_rate = torchaudio.load(audio_file)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        result = pipeline(audio_input)

        # pyannote 4.x returns DiarizeOutput; 3.x returns Annotation directly
        if hasattr(result, 'speaker_diarization'):
            diarization = result.speaker_diarization
        else:
            diarization = result

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker_label': speaker
            })
        
        return segments
    except Exception as e:
        raise RuntimeError(f"Diarization failed: {str(e)}")


def format_diarization_output(segments: List[Dict]) -> str:
    """
    Format diarization segments as human-readable string.
    
    Args:
        segments: List of diarization segments
        
    Returns:
        Formatted string
    """
    output = []
    for seg in segments:
        output.append(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker_label']}")
    return "\n".join(output)


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diarize.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    print(f"Diarizing {audio_file}...")
    
    segments = diarize_audio(audio_file)
    print(f"\nFound {len(segments)} speaker segments:")
    print(format_diarization_output(segments))
