"""
Unified pipeline: diarization (pyannote) -> ASR (WhisperX) -> alignment ->
per-segment spoof detection + open-set speaker recognition.
"""

import argparse
from datetime import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch

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
from embeddings.speaker_id import (
    extract_embedding_from_segment,
    load_speaker_encoder,
    load_speaker_registry,
    resolve_speaker_display,
)
from spoof_detection.infer import SpoofDetector


def load_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    os.environ["HF_TOKEN"] = token
                    return token
    return None


def default_registry_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "speaker_registry.json")


def output_json_path(audio_file: str) -> str:
    base = os.path.splitext(os.path.basename(audio_file))[0]
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    os.makedirs(results_dir, exist_ok=True)

    return os.path.join(results_dir, base + ".json")


def build_summary(segments: List[Dict[str, Any]]) -> Dict[str, int]:
    speakers = {s["speaker"] for s in segments}
    return {
        "total_speakers": len(speakers),
        "real_segments": sum(1 for s in segments if s["label"] == "REAL"),
        "spoof_segments": sum(1 for s in segments if s["label"] == "SPOOF"),
        "authorized_segments": sum(1 for s in segments if s["type"] == "AUTHORIZED"),
        "unauthorized_segments": sum(1 for s in segments if s["type"] == "UNAUTHORIZED"),
    }


def run_pipeline(
    audio_file: str,
    hf_token: Optional[str] = None,
    whisper_model: str = "base",
    speaker_threshold: float = 0.75,
    merge_gap: float = 0.5,
    registry_path: Optional[str] = None,
) -> Dict[str, Any]:
    registry_path = registry_path or default_registry_path()





    print("=" * 60)
    print("Speaker Diarization & Secure Recognition Pipeline")
    print("=" * 60)
    print(f"Input: {audio_file}\n")
    print("[1/5] Preprocessing audio...")
    audio, sr = load_audio(audio_file, target_sr=16000, mono=True)
    print("[2/5] Speaker diarization (pyannote)...")
    diarization_pipeline = load_diarization_pipeline(use_auth_token=hf_token)
    diarization_segments = diarize_audio(
        audio_file, pipeline=diarization_pipeline, use_auth_token=hf_token
    )
    print("[3/5] Speech recognition (WhisperX)...")
    whisperx_models = load_whisperx_model(whisper_model)
    asr_segments = transcribe_audio(
        audio_file, models=whisperx_models, model_name=whisper_model
    )
    print("[4/5] Aligning speakers with text...")
    aligned_segments = align_segments(asr_segments, diarization_segments)
    aligned_segments = merge_adjacent_segments(
        aligned_segments, same_speaker_threshold=merge_gap
    )

    print("[5/5] Spoof detection & speaker identification...")
    detector = SpoofDetector()
    encoder = load_speaker_encoder()
    registry = load_speaker_registry(registry_path)

    dia_to_guest: Dict[str, str] = {}
    guest_counter = [0]
    min_embed_samples = int(sr * 0.5)

    final_segments: List[Dict[str, Any]] = []
    for seg in aligned_segments:
        t_start = float(seg.get("start", 0.0))
        t_end = float(seg.get("end", 0.0))
        dia_label = str(seg.get("speaker_label") or "UNKNOWN")

        i_start = int(t_start * sr)
        i_end = min(int(t_end * sr), len(audio))
        seg_audio = audio[i_start:i_end]

        if len(seg_audio) > 0:
            spoof_name, _score = detector.predict(seg_audio, sr=16000)
            label = "SPOOF" if spoof_name.lower() == "spoof" else "REAL"
        else:
            label = "REAL"

        embedding = None
        if len(seg_audio) >= min_embed_samples:
            try:
                embedding = extract_embedding_from_segment(seg_audio, sr, encoder)
            except Exception:
                embedding = None
        # # DEBUG STATEMENT
        # print(f"Segment {t_start}-{t_end}, len={len(seg_audio)}, embedding={embedding is not None}")

        speaker_name, seg_type = resolve_speaker_display(
            dia_label,
            embedding,
            registry,
            speaker_threshold,
            dia_to_guest,
            guest_counter,
        )

        final_segments.append(
            {
                "start": round(t_start, 2),
                "end": round(t_end, 2),
                "speaker": speaker_name,
                "type": seg_type,
                "label": label,
                "text": seg.get("text", "") or "",
            }
        )

    summary = build_summary(final_segments)
    # out: Dict[str, Any] = {
    #     "audio_file": os.path.basename(audio_file),
    #     "summary": summary,
    #     "segments": final_segments,
    # }
    out = {
    "audio_file": os.path.basename(audio_file),
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "speaker_threshold": speaker_threshold,
        "merge_gap": merge_gap,
        "whisper_model": whisper_model
    },
    "summary": summary,
    "segments": final_segments
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified diarization, ASR, spoof detection, and speaker recognition."
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to input audio file",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for pyannote (else HF_TOKEN / .env)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--speaker-threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for registry match (default: 0.75)",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.5,
        help="Max gap in seconds to merge adjacent segments (default: 0.5)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to speaker registry JSON (default: data/speaker_registry.json)",
    )

    args = parser.parse_args()
    audio_file = os.path.abspath(args.audio_file)

    if not os.path.isfile(audio_file):
        print(f"Error: Audio file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)

    out_path = output_json_path(audio_file)

    try:
        result = run_pipeline(
            audio_file=audio_file,
            hf_token=args.hf_token or load_hf_token(),
            whisper_model=args.whisper_model,
            speaker_threshold=args.speaker_threshold,
            merge_gap=args.merge_gap,
            registry_path=args.registry,
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Done. Wrote {out_path}")
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
