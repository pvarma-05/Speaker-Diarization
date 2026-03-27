"""
Main CLI entry point for S-O-EEND-SDR pipeline.
Orchestrates the entire speaker diarization and recognition workflow.
"""

import argparse
import json
import sys
import os
from typing import Dict, List

# torch >=2.6 defaults weights_only=True in torch.load, which breaks
# pyannote model loading. Force weights_only=False globally.
import torch
_orig_load = torch.serialization.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
torch.serialization.load = _patched_load

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.audio_utils import load_audio
from diarization.diarize import diarize_audio, load_diarization_pipeline
from asr.transcribe import transcribe_audio, load_whisperx_model
from alignment.align import align_segments, merge_adjacent_segments
from embeddings.speaker_id import identify_speakers_in_segments


def load_hf_token():
    """Load HuggingFace token from .env file or environment variable."""
    # Check environment variable first
    token = os.environ.get('HF_TOKEN')
    if token:
        return token
    # Try loading from .env file in project root
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('HF_TOKEN='):
                    token = line.split('=', 1)[1].strip()
                    os.environ['HF_TOKEN'] = token  # Set for downstream libraries
                    return token
    return None


def format_final_output(segments: List[Dict], mode: str = "auth") -> str:
    """
    Format final output for CLI display.

    Args:
        segments: List of final segments with all information
        mode: Pipeline mode ("auth" or "deepfake")

    Returns:
        Formatted string
    """
    output = []
    for seg in segments:
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        speaker = seg.get('identified_speaker', seg.get('speaker_label', 'UNKNOWN'))
        text = seg.get('text', '')

        if mode == "deepfake":
            spoof_label = seg.get('spoof_label', 'Unknown')
            spoof_score = seg.get('spoof_score', 0.0)
            if spoof_label == "Spoof":
                flag = f"[!!] AI/TTS {spoof_score:.0%}"
            else:
                flag = f"[OK] Human {spoof_score:.0%}"
            output.append(f"[{start:.2f}-{end:.2f}] {speaker}: {text} ({flag})")
        else:
            similarity = seg.get('similarity_score', 0.0)
            is_authorized = speaker != 'UNKNOWN'
            auth_flag = "[OK]" if is_authorized else "[!!] UNAUTHORIZED"
            output.append(f"[{start:.2f}-{end:.2f}] {speaker}: {text} ({auth_flag} {similarity:.0%})")

    return "\n".join(output)


def run_pipeline(audio_file: str, output_json: str = None,
                hf_token: str = None, whisper_model: str = "base",
                speaker_threshold: float = 0.75, merge_gap: float = 0.5,
                enable_auth_filter: bool = True,
                registry_path: str = None,
                mode: str = "auth") -> Dict:
    """
    Run the complete S-O-EEND-SDR pipeline.
    
    Args:
        audio_file: Path to input audio file
        output_json: Optional path to save JSON output
        hf_token: Optional HuggingFace token for model access
        whisper_model: Whisper model size (tiny, base, small, medium, large)
        speaker_threshold: Similarity threshold for speaker identification
        merge_gap: Maximum gap in seconds to merge adjacent segments
        enable_auth_filter: Whether to filter out unauthorized speakers
        
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
        print(f"  [OK] Loaded audio: {len(audio)/sr:.2f} seconds, {sr} Hz\n")
    except Exception as e:
        print(f"  [!!] Error: {str(e)}")
        sys.exit(1)
    
    # Step 2: Speaker Diarization
    print("Step 2: Speaker diarization...")
    try:
        diarization_pipeline = load_diarization_pipeline(use_auth_token=hf_token)
        diarization_segments = diarize_audio(audio_file, pipeline=diarization_pipeline, 
                                            use_auth_token=hf_token)
        print(f"  [OK] Found {len(diarization_segments)} speaker segments\n")
    except Exception as e:
        print(f"  [!!] Error: {str(e)}")
        print("  Note: You may need to accept model terms at:")
        print("  https://huggingface.co/pyannote/speaker-diarization-3.1")
        sys.exit(1)
    
    # Step 3: Automatic Speech Recognition
    print("Step 3: Automatic speech recognition...")
    try:
        whisperx_models = load_whisperx_model(whisper_model)
        asr_segments = transcribe_audio(audio_file, models=whisperx_models, 
                                       model_name=whisper_model)
        print(f"  [OK] Generated {len(asr_segments)} transcription segments\n")
    except Exception as e:
        print(f"  [!!] Error: {str(e)}")
        sys.exit(1)
    
    # Step 4: Alignment & Fusion
    print("Step 4: Aligning ASR with diarization...")
    try:
        aligned_segments = align_segments(asr_segments, diarization_segments)
        aligned_segments = merge_adjacent_segments(aligned_segments, 
                                                  same_speaker_threshold=merge_gap)
        print(f"  [OK] Aligned {len(aligned_segments)} segments\n")
    except Exception as e:
        print(f"  [!!] Error: {str(e)}")
        sys.exit(1)
    
    if mode == "auth":
        # Step 5: Speaker Identification (Open-Set)
        print("Step 5: Speaker identification (open-set)...")
        try:
            id_kwargs = {"threshold": speaker_threshold}
            if registry_path:
                id_kwargs["registry_path"] = registry_path
            identified_segments = identify_speakers_in_segments(
                aligned_segments, audio_file, **id_kwargs
            )
            print(f"  [OK] Identified speakers for {len(identified_segments)} segments\n")
        except Exception as e:
            print(f"  [!!] Warning: Speaker identification failed: {str(e)}")
            print("  Continuing with diarization labels only...")
            identified_segments = aligned_segments
            for seg in identified_segments:
                seg['identified_speaker'] = seg.get('speaker_label', 'UNKNOWN')
                seg['similarity_score'] = 0.0

        # Step 6: Speaker Authorization
        if enable_auth_filter:
            print("Step 6: Speaker authorization filter...")
            total_segments = len(identified_segments)
            authorized_segments = [s for s in identified_segments if s.get('identified_speaker', 'UNKNOWN') != 'UNKNOWN']
            unauthorized_count = total_segments - len(authorized_segments)

            if unauthorized_count > 0:
                unauthorized_labels = set(
                    s.get('speaker_label', '?') for s in identified_segments
                    if s.get('identified_speaker', 'UNKNOWN') == 'UNKNOWN'
                )
                print(f"  [OK] Authorized {len(authorized_segments)}/{total_segments} segments")
                print(f"  [!!] Filtered out {unauthorized_count} segments from {len(unauthorized_labels)} unauthorized speaker(s)\n")
                identified_segments = authorized_segments
            else:
                print(f"  [OK] All {total_segments} segments from authorized speakers\n")
        else:
            print("Step 6: Authorization filter disabled - all speakers included.\n")

    elif mode == "deepfake":
        # Step 5: Spoof Detection (skip speaker identification entirely)
        print("Step 5: Deepfake / spoof detection (wav2vec2)...")
        try:
            from spoof_detection.infer import SpoofDetector

            print("  Loading wav2vec2 deepfake detector...")
            detector = SpoofDetector()
            print("  Model loaded")

            real_count = 0
            spoof_count = 0

            # Build a mapping from diarization labels to human-friendly names
            # Only real (human) speakers get numbered; spoof segments are labeled "Spoof"
            diarization_label_to_speaker = {}
            speaker_counter = 0

            identified_segments = []
            for seg in aligned_segments:
                new_seg = seg.copy()
                t_start = seg.get('start', 0.0)
                t_end = seg.get('end', 0.0)

                # Extract audio segment
                i_start = int(t_start * sr)
                i_end = min(int(t_end * sr), len(audio))
                seg_audio = audio[i_start:i_end]

                if len(seg_audio) > 0:
                    label, score = detector.predict(seg_audio, sr=16000)
                    new_seg['spoof_label'] = label
                    new_seg['spoof_score'] = score

                    if label == "Spoof":
                        new_seg['identified_speaker'] = "Spoof"
                        spoof_count += 1
                        print(f"  ! Spoof detected: [{t_start:.1f}-{t_end:.1f}s] (confidence: {score:.2f})")
                    else:
                        # Map diarization label to a human-friendly speaker number
                        dia_label = seg.get('speaker_label', 'UNKNOWN')
                        if dia_label not in diarization_label_to_speaker:
                            speaker_counter += 1
                            diarization_label_to_speaker[dia_label] = f"Speaker {speaker_counter}"
                        new_seg['identified_speaker'] = diarization_label_to_speaker[dia_label]
                        new_seg['similarity_score'] = score
                        real_count += 1
                else:
                    new_seg['identified_speaker'] = seg.get('speaker_label', 'UNKNOWN')
                    new_seg['spoof_label'] = 'Unknown'
                    new_seg['spoof_score'] = 0.0

                identified_segments.append(new_seg)

            print(f"\n  [OK] Analysis complete: {real_count} human segments, {spoof_count} spoof/AI segments")
            if diarization_label_to_speaker:
                print(f"  [OK] Identified {len(diarization_label_to_speaker)} unique human speaker(s)\n")
            else:
                print()

        except Exception as e:
            print(f"  [!!] Error during deepfake detection: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fall back to diarization labels
            identified_segments = aligned_segments
            for seg in identified_segments:
                seg['identified_speaker'] = seg.get('speaker_label', 'UNKNOWN')
    
    # Step 7: Final Output
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(format_final_output(identified_segments, mode=mode))
    print()
    
    # Prepare JSON output
    json_output = {
        'audio_file': audio_file,
        'num_segments': len(identified_segments),
        'segments': identified_segments
    }
    
    # Save JSON if requested
    if output_json:
        os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
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
  python main.py audio.wav --hf-token YOUR_TOKEN --no-auth-filter
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
    parser.add_argument('--speaker-threshold', type=float, default=0.75,
                       help='Similarity threshold for speaker identification (default: 0.75)')
    parser.add_argument('--merge-gap', type=float, default=0.5,
                       help='Maximum gap in seconds to merge adjacent segments (default: 0.5)')
    parser.add_argument('--no-auth-filter', action='store_true',
                        help='Disable speaker authorization filter (include all speakers)')
    parser.add_argument('--registry', type=str, default=None,
                       help='Path to speaker registry JSON (default: data/speaker_registry.json)')
    parser.add_argument('--mode', type=str, default='auth', choices=['auth', 'deepfake'],
                       help='Pipeline mode: "auth" (default) for speaker authorization, "deepfake" for anti-spoofing analysis')
    
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
            hf_token=args.hf_token or load_hf_token(),
            whisper_model=args.whisper_model,
            speaker_threshold=args.speaker_threshold,
            merge_gap=args.merge_gap,
            enable_auth_filter=not args.no_auth_filter,
            registry_path=args.registry,
            mode=args.mode
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
