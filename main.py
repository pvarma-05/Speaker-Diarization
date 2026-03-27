"""
Main CLI entry point for the Speaker Diarization & Secure Recognition pipeline.
Orchestrates: Preprocessing -> Diarization -> ASR -> Alignment -> Mode Switch -> Output
"""

import argparse
import json
import sys
import os
from typing import Dict, List

import torch

# Torch >=2.6 defaults weights_only=True which breaks pyannote model loading.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.audio_utils import load_audio
from diarization.diarize import diarize_audio, load_diarization_pipeline
from asr.transcribe import transcribe_audio, load_whisperx_model
from alignment.align import align_segments, merge_adjacent_segments
from embeddings.speaker_id import identify_speakers_in_segments


def load_hf_token():
    """Load HuggingFace token from .env file or environment variable."""
    token = os.environ.get('HF_TOKEN')
    if token:
        return token
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('HF_TOKEN='):
                    token = line.split('=', 1)[1].strip()
                    os.environ['HF_TOKEN'] = token
                    return token
    return None


def format_final_output(segments: List[Dict], mode: str = "auth") -> str:
    """Format final segments for CLI display."""
    lines = []
    for seg in segments:
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        speaker = seg.get('speaker', 'UNKNOWN')
        text = seg.get('text', '')
        label = seg.get('label', 'UNKNOWN')

        if mode == "deepfake":
            flag = "[!!] SPOOF" if label == "SPOOF" else "[OK] REAL"
        else:
            flag = "[OK]" if label != "UNKNOWN" else "[!!] UNKNOWN"

        lines.append(f"[{start:.2f}-{end:.2f}] {speaker}: {text}  ({flag})")

    return "\n".join(lines)


def run_pipeline(audio_file: str, output_json: str = None,
                 hf_token: str = None, whisper_model: str = "base",
                 speaker_threshold: float = 0.75, merge_gap: float = 0.5,
                 enable_auth_filter: bool = True,
                 registry_path: str = None,
                 mode: str = "auth") -> Dict:
    """
    Run the complete pipeline.

    Args:
        audio_file: Path to input audio file
        output_json: Optional path to save JSON output
        hf_token: HuggingFace token for pyannote model access
        whisper_model: Whisper model size (tiny, base, small, medium, large)
        speaker_threshold: Cosine similarity threshold for speaker identification
        merge_gap: Maximum gap in seconds to merge adjacent segments
        enable_auth_filter: Whether to filter out unauthorized speakers
        registry_path: Path to speaker registry JSON
        mode: Pipeline mode - "auth" or "deepfake"

    Returns:
        Dictionary with pipeline results
    """
    print("=" * 60)
    print("Speaker Diarization & Secure Recognition Pipeline")
    print(f"Mode: {'Spoof Detection' if mode == 'deepfake' else 'Open-Set Recognition'}")
    print("=" * 60)
    print(f"Input: {audio_file}\n")

    # Step 1: Preprocessing
    print("[1/5] Preprocessing audio...")
    try:
        audio, sr = load_audio(audio_file, target_sr=16000, mono=True)
        print(f"  OK - {len(audio)/sr:.2f}s, {sr} Hz\n")
    except Exception as e:
        print(f"  FAIL - {e}")
        sys.exit(1)

    # Step 2: Speaker Diarization (pyannote)
    print("[2/5] Speaker diarization (pyannote)...")
    try:
        diarization_pipeline = load_diarization_pipeline(use_auth_token=hf_token)
        diarization_segments = diarize_audio(
            audio_file, pipeline=diarization_pipeline, use_auth_token=hf_token
        )
        print(f"  OK - {len(diarization_segments)} segments\n")
    except Exception as e:
        print(f"  FAIL - {e}")
        print("  You may need to accept model terms at:")
        print("  https://huggingface.co/pyannote/speaker-diarization-3.1")
        sys.exit(1)

    # Step 3: ASR (WhisperX)
    print("[3/5] Speech recognition (WhisperX)...")
    try:
        whisperx_models = load_whisperx_model(whisper_model)
        asr_segments = transcribe_audio(
            audio_file, models=whisperx_models, model_name=whisper_model
        )
        print(f"  OK - {len(asr_segments)} segments\n")
    except Exception as e:
        print(f"  FAIL - {e}")
        sys.exit(1)

    # Step 4: Alignment
    print("[4/5] Aligning speakers with text...")
    try:
        aligned_segments = align_segments(asr_segments, diarization_segments)
        aligned_segments = merge_adjacent_segments(
            aligned_segments, same_speaker_threshold=merge_gap
        )
        print(f"  OK - {len(aligned_segments)} aligned segments\n")
    except Exception as e:
        print(f"  FAIL - {e}")
        sys.exit(1)

    # Step 5: Mode-specific processing
    final_segments = []

    if mode == "deepfake":
        print("[5/5] Spoof detection (Wav2Vec2)...")
        try:
            from spoof_detection.infer import SpoofDetector
            detector = SpoofDetector()

            label_map = {}
            speaker_counter = 0

            for seg in aligned_segments:
                t_start = seg.get('start', 0.0)
                t_end = seg.get('end', 0.0)

                i_start = int(t_start * sr)
                i_end = min(int(t_end * sr), len(audio))
                seg_audio = audio[i_start:i_end]

                dia_label = seg.get('speaker_label', 'UNKNOWN')
                if dia_label not in label_map:
                    speaker_counter += 1
                    label_map[dia_label] = f"Speaker_{speaker_counter}"
                speaker_name = label_map[dia_label]

                if len(seg_audio) > 0:
                    spoof_label, score = detector.predict(seg_audio, sr=16000)
                    final_segments.append({
                        'start': round(t_start, 2),
                        'end': round(t_end, 2),
                        'speaker': speaker_name,
                        'text': seg.get('text', ''),
                        'label': spoof_label.upper()
                    })
                else:
                    final_segments.append({
                        'start': round(t_start, 2),
                        'end': round(t_end, 2),
                        'speaker': speaker_name,
                        'text': seg.get('text', ''),
                        'label': 'UNKNOWN'
                    })

            real_count = sum(1 for s in final_segments if s['label'] == 'REAL')
            spoof_count = sum(1 for s in final_segments if s['label'] == 'SPOOF')
            print(f"  OK - {real_count} real, {spoof_count} spoof\n")

        except Exception as e:
            print(f"  FAIL - {e}")
            import traceback
            traceback.print_exc()
            for seg in aligned_segments:
                final_segments.append({
                    'start': round(seg.get('start', 0.0), 2),
                    'end': round(seg.get('end', 0.0), 2),
                    'speaker': seg.get('speaker_label', 'UNKNOWN'),
                    'text': seg.get('text', ''),
                    'label': 'UNKNOWN'
                })

    elif mode == "auth":
        print("[5/5] Speaker identification (ECAPA-TDNN)...")
        try:
            id_kwargs = {"threshold": speaker_threshold}
            if registry_path:
                id_kwargs["registry_path"] = registry_path

            identified = identify_speakers_in_segments(
                aligned_segments, audio_file, **id_kwargs
            )

            for seg in identified:
                speaker_name = seg.get('identified_speaker', 'UNKNOWN')
                final_segments.append({
                    'start': round(seg.get('start', 0.0), 2),
                    'end': round(seg.get('end', 0.0), 2),
                    'speaker': speaker_name,
                    'text': seg.get('text', ''),
                    'label': speaker_name
                })

            known = sum(1 for s in final_segments if s['label'] != 'UNKNOWN')
            unknown = len(final_segments) - known
            print(f"  OK - {known} identified, {unknown} unknown\n")

        except Exception as e:
            print(f"  FAIL - Speaker identification failed: {e}")
            print("  Continuing with diarization labels...")
            for seg in aligned_segments:
                final_segments.append({
                    'start': round(seg.get('start', 0.0), 2),
                    'end': round(seg.get('end', 0.0), 2),
                    'speaker': seg.get('speaker_label', 'UNKNOWN'),
                    'text': seg.get('text', ''),
                    'label': 'UNKNOWN'
                })

        if enable_auth_filter:
            before = len(final_segments)
            final_segments = [s for s in final_segments if s['label'] != 'UNKNOWN']
            filtered = before - len(final_segments)
            if filtered > 0:
                print(f"  Auth filter: removed {filtered}/{before} unknown segments\n")

    # Final Output
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(format_final_output(final_segments, mode=mode))
    print()

    json_output = {
        'audio_file': audio_file,
        'mode': mode,
        'num_segments': len(final_segments),
        'segments': final_segments
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"Saved to: {output_json}")

    return json_output


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Speaker Diarization & Secure Speaker Recognition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --audio input.wav --mode deepfake
  python main.py --audio input.wav --mode auth
  python main.py --audio input.wav --mode deepfake --output results/output.json
  python main.py --audio input.wav --mode auth --speaker-threshold 0.8
        """
    )

    parser.add_argument('--audio', type=str, required=True,
                        help='Path to input audio file')
    parser.add_argument('--mode', type=str, default='auth',
                        choices=['auth', 'deepfake'],
                        help='Pipeline mode: "auth" (speaker recognition) '
                             'or "deepfake" (spoof detection)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save JSON output')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='HuggingFace token for pyannote model access')
    parser.add_argument('--whisper-model', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: base)')
    parser.add_argument('--speaker-threshold', type=float, default=0.75,
                        help='Cosine similarity threshold for speaker ID (default: 0.75)')
    parser.add_argument('--merge-gap', type=float, default=0.5,
                        help='Max gap in seconds to merge adjacent segments (default: 0.5)')
    parser.add_argument('--no-auth-filter', action='store_true',
                        help='Disable speaker authorization filter')
    parser.add_argument('--registry', type=str, default=None,
                        help='Path to speaker registry JSON '
                             '(default: data/speaker_registry.json)')

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    try:
        run_pipeline(
            audio_file=args.audio,
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
        print("\nPipeline interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
