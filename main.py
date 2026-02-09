"""
Main CLI entry point for S-O-EEND-SDR pipeline.
Orchestrates the entire speaker diarization and recognition workflow.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.audio_utils import load_audio
from diarization.diarize import diarize_audio, load_diarization_pipeline
from asr.transcribe import transcribe_audio, load_whisperx_model
from alignment.align import align_segments, merge_adjacent_segments
from embeddings.speaker_id import identify_speakers_in_segments
from spoofing.spoof_check import check_spoofing_batch


def format_final_output(segments: List[Dict]) -> str:
    """
    Format final output for CLI display.
    
    Args:
        segments: List of final segments with all information
        
    Returns:
        Formatted string
    """
    output = []
    for seg in segments:
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        speaker = seg.get('identified_speaker', seg.get('speaker_label', 'UNKNOWN'))
        text = seg.get('text', '')
        
        # Get spoofing status
        spoof_check = seg.get('spoof_check', {})
        is_genuine = spoof_check.get('is_genuine', True)
        spoof_flag = "✓" if is_genuine else "✗"
        
        output.append(f"[{start:.2f}-{end:.2f}] {speaker}: {text} ({spoof_flag})")
    
    return "\n".join(output)


def run_pipeline(audio_file: str, output_json: str = None, 
                hf_token: str = None, whisper_model: str = "base",
                speaker_threshold: float = 0.7, merge_gap: float = 0.5,
                enable_spoof_check: bool = True) -> Dict:
    """
    Run the complete S-O-EEND-SDR pipeline.
    
    Args:
        audio_file: Path to input audio file
        output_json: Optional path to save JSON output
        hf_token: Optional HuggingFace token for model access
        whisper_model: Whisper model size (tiny, base, small, medium, large)
        speaker_threshold: Similarity threshold for speaker identification
        merge_gap: Maximum gap in seconds to merge adjacent segments
        enable_spoof_check: Whether to run spoofing detection
        
    Returns:
        Dictionary with complete pipeline results
    """
    print("=" * 60)
    print("S-O-EEND-SDR Pipeline")
    print("=" * 60)
    print(f"Input audio: {audio_file}\n")
    
    # Step 1: Audio Input & Preprocessing
    print("Step 1: Loading and preprocessing audio...")
    try:
        audio, sr = load_audio(audio_file, target_sr=16000, mono=True)
        print(f"  ✓ Loaded audio: {len(audio)/sr:.2f} seconds, {sr} Hz\n")
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        sys.exit(1)
    
    # Step 2: Speaker Diarization
    print("Step 2: Speaker diarization...")
    try:
        diarization_pipeline = load_diarization_pipeline(use_auth_token=hf_token)
        diarization_segments = diarize_audio(audio_file, pipeline=diarization_pipeline, 
                                            use_auth_token=hf_token)
        print(f"  ✓ Found {len(diarization_segments)} speaker segments\n")
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        print("  Note: You may need to accept model terms at:")
        print("  https://huggingface.co/pyannote/speaker-diarization-3.1")
        sys.exit(1)
    
    # Step 3: Automatic Speech Recognition
    print("Step 3: Automatic speech recognition...")
    try:
        whisperx_models = load_whisperx_model(whisper_model)
        asr_segments = transcribe_audio(audio_file, models=whisperx_models, 
                                       model_name=whisper_model)
        print(f"  ✓ Generated {len(asr_segments)} transcription segments\n")
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        sys.exit(1)
    
    # Step 4: Alignment & Fusion
    print("Step 4: Aligning ASR with diarization...")
    try:
        aligned_segments = align_segments(asr_segments, diarization_segments)
        aligned_segments = merge_adjacent_segments(aligned_segments, 
                                                  same_speaker_threshold=merge_gap)
        print(f"  ✓ Aligned {len(aligned_segments)} segments\n")
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        sys.exit(1)
    
    # Step 5: Speaker Identification (Open-Set)
    print("Step 5: Speaker identification (open-set)...")
    try:
        identified_segments = identify_speakers_in_segments(
            aligned_segments, audio_file, threshold=speaker_threshold
        )
        print(f"  ✓ Identified speakers for {len(identified_segments)} segments\n")
    except Exception as e:
        print(f"  ✗ Warning: Speaker identification failed: {str(e)}")
        print("  Continuing with diarization labels only...")
        identified_segments = aligned_segments
        for seg in identified_segments:
            seg['identified_speaker'] = seg.get('speaker_label', 'UNKNOWN')
            seg['similarity_score'] = 0.0
    
    # Step 6: Anti-Spoofing (Optional)
    if enable_spoof_check:
        print("Step 6: Anti-spoofing detection...")
        try:
            identified_segments = check_spoofing_batch(identified_segments, audio_file)
            print(f"  ✓ Checked {len(identified_segments)} segments (placeholder)\n")
        except Exception as e:
            print(f"  ✗ Warning: Spoofing check failed: {str(e)}")
            print("  Continuing without spoofing detection...")
            for seg in identified_segments:
                seg['spoof_check'] = {'is_genuine': True, 'confidence': 1.0, 'method': 'skipped'}
    else:
        for seg in identified_segments:
            seg['spoof_check'] = {'is_genuine': True, 'confidence': 1.0, 'method': 'disabled'}
    
    # Step 7: Final Output
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(format_final_output(identified_segments))
    print()
    
    # Prepare JSON output
    json_output = {
        'audio_file': audio_file,
        'num_segments': len(identified_segments),
        'segments': identified_segments
    }
    
    # Save JSON if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"Results saved to: {output_json}")
    
    return json_output


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Secure, Open-Set, Near End-to-End Speaker Diarization and Recognition (S-O-EEND-SDR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py audio.wav
  python main.py audio.wav --output results.json
  python main.py audio.wav --whisper-model large --speaker-threshold 0.8
  python main.py audio.wav --hf-token YOUR_TOKEN --no-spoof-check
        """
    )
    
    parser.add_argument('audio_file', type=str, help='Path to input audio file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save JSON output file')
    parser.add_argument('--hf-token', type=str, default=None,
                       help='HuggingFace token for model access')
    parser.add_argument('--whisper-model', type=str, default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: base)')
    parser.add_argument('--speaker-threshold', type=float, default=0.7,
                       help='Similarity threshold for speaker identification (default: 0.7)')
    parser.add_argument('--merge-gap', type=float, default=0.5,
                       help='Maximum gap in seconds to merge adjacent segments (default: 0.5)')
    parser.add_argument('--no-spoof-check', action='store_true',
                       help='Disable spoofing detection')
    
    args = parser.parse_args()
    
    # Validate audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Run pipeline
    try:
        run_pipeline(
            audio_file=args.audio_file,
            output_json=args.output,
            hf_token=args.hf_token,
            whisper_model=args.whisper_model,
            speaker_threshold=args.speaker_threshold,
            merge_gap=args.merge_gap,
            enable_spoof_check=not args.no_spoof_check
        )
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
