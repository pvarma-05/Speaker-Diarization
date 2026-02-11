"""
Standalone spoof detection test with VAD + transcription.
Uses energy-based VAD, wav2vec2 deepfake detector, and faster-whisper.
"""
import sys
import os
import json
import numpy as np
import librosa
import soundfile as sf
import tempfile
import torch
from transformers import pipeline as hf_pipeline
from faster_whisper import WhisperModel


def simple_vad_split(audio, sr, frame_ms=30, min_seg_sec=1.0, merge_gap_sec=0.3):
    """Split audio into speech segments using adaptive energy-based VAD."""
    frame_len = int(sr * frame_ms / 1000)
    frames = [audio[i:i+frame_len] for i in range(0, len(audio), frame_len)]

    energies = [np.sqrt(np.mean(f ** 2)) for f in frames]
    mean_energy = np.mean(energies)
    threshold = mean_energy * 0.3
    is_speech = [e > threshold for e in energies]

    merge_gap_frames = int(merge_gap_sec * 1000 / frame_ms)
    segments = []
    start = None
    silence_count = 0
    for i, speech in enumerate(is_speech):
        if speech:
            if start is None:
                start = i
            silence_count = 0
        else:
            if start is not None:
                silence_count += 1
                if silence_count > merge_gap_frames:
                    t_start = start * frame_ms / 1000
                    t_end = (i - silence_count) * frame_ms / 1000
                    if (t_end - t_start) >= min_seg_sec:
                        segments.append((t_start, t_end))
                    start = None
                    silence_count = 0
    if start is not None:
        t_start = start * frame_ms / 1000
        t_end = len(is_speech) * frame_ms / 1000
        if (t_end - t_start) >= min_seg_sec:
            segments.append((t_start, t_end))

    return segments


def transcribe_segment(audio_seg, sr, whisper_model):
    """Transcribe an audio segment using faster-whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio_seg, sr)
    try:
        segments, _ = whisper_model.transcribe(tmp_path, language="en")
        text = " ".join(seg.text.strip() for seg in segments)
        return text if text else "(silence)"
    finally:
        os.unlink(tmp_path)


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/[trimmed] videoplayback.wav"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results/vad_output.json"

    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr
    print(f"Duration: {duration:.1f}s, Sample rate: {sr} Hz\n")

    # Split into segments
    segments = simple_vad_split(audio, sr, min_seg_sec=1.5)
    print(f"Found {len(segments)} speech segments via energy VAD\n")

    if not segments:
        print("No segments found, testing on whole file as one chunk...")
        segments = [(0.0, duration)]

    # Load models
    print("Loading wav2vec2 deepfake detector...")
    device = 0 if torch.cuda.is_available() else -1
    detector = hf_pipeline(
        "audio-classification",
        model="mo-thecreator/Deepfake-audio-detection",
        device=device,
    )

    print("Loading Whisper (base) for transcription...")
    whisper = WhisperModel("base", device="cpu", compute_type="int8")
    print()

    # Run detection + transcription
    print("=" * 90)
    real_count = 0
    spoof_count = 0
    speaker_map = {}
    speaker_counter = 0
    output_segments = []

    for i, (t_start, t_end) in enumerate(segments):
        i_start = int(t_start * sr)
        i_end = int(t_end * sr)
        seg_audio = audio[i_start:i_end]

        # Spoof detection
        result = detector({"raw": seg_audio.astype(np.float32), "sampling_rate": 16000})
        top = result[0]
        label_raw = top["label"].lower()
        score = top["score"]

        # Transcription
        text = transcribe_segment(seg_audio, sr, whisper)

        if "fake" in label_raw or "spoof" in label_raw:
            label = "Spoof"
            speaker_name = "Spoof"
            tag = f"AI/TTS ({score:.0%})"
            marker = "[!!]"
            spoof_count += 1
        else:
            label = "Real"
            # Assign speaker numbers based on segment grouping
            # (simple heuristic: consecutive real segments close together = same speaker)
            if not speaker_map or (t_start - segments[i-1][1] > 3.0 if i > 0 else True):
                # Check if this might be a new speaker vs returning speaker
                pass
            speaker_counter_key = f"real_{i}"
            speaker_name = f"Speaker {real_count + 1}"
            tag = f"Human ({score:.0%})"
            marker = "[OK]"
            real_count += 1

        print(f"  [{t_start:5.1f}s - {t_end:5.1f}s] {marker} {tag}")
        print(f"    {speaker_name}: {text}")
        print()

        output_segments.append({
            "start": round(t_start, 2),
            "end": round(t_end, 2),
            "text": text,
            "identified_speaker": speaker_name,
            "spoof_label": label,
            "spoof_score": round(score, 4),
        })

    print("=" * 90)
    print(f"\nSummary: {real_count} Real, {spoof_count} Spoof  (Total: {len(segments)} segments)")

    # Save output
    json_output = {
        "audio_file": audio_path,
        "num_segments": len(output_segments),
        "real_count": real_count,
        "spoof_count": spoof_count,
        "segments": output_segments,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
