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
        use_auth_token: HuggingFace token if needed for model access
        
    Returns:
        pyannote.audio Pipeline object
    """
    try:
        # Use the pretrained diarization pipeline
        # Note: User needs to accept model terms on HuggingFace
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
        # Run diarization
        diarization = pipeline(audio_file)
        
        # Convert to list of dictionaries
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
