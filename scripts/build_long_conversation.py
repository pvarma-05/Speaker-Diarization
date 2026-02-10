"""
Build a longer (~5 min) mixed conversation from LibriSpeech clips
and run the full pipeline on it.

Uses all 12 clips per speaker to create a realistic back-and-forth
conversation with 5 speakers (3 authorized, 2 unauthorized).
"""

import os
import sys
import random
import numpy as np
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BENCH_DIR = os.path.join(DATA_DIR, "benchmark_enhanced")
OUTPUT_WAV = os.path.join(DATA_DIR, "long_conversation.wav")

# 3 authorized + 2 unauthorized = 5 speakers in the conversation
SPEAKERS = {
    "Alice":   "authorized",
    "Bob":     "authorized",
    "Charlie": "authorized",
    "Frank":   "unauthorized",
    "Grace":   "unauthorized",
}

SR = 16000  # target sample rate


def build_conversation():
    """Build a ~5 min conversation by interleaving clips from 5 speakers."""
    print("Building long conversation...")
    print(f"  Speakers: {', '.join(f'{n} ({s})' for n, s in SPEAKERS.items())}")

    # Collect all clips per speaker
    speaker_clips = {}
    for name in SPEAKERS:
        spk_dir = os.path.join(BENCH_DIR, name)
        if not os.path.exists(spk_dir):
            print(f"  Warning: No directory for {name}")
            continue
        wavs = sorted([f for f in os.listdir(spk_dir) if f.endswith(".wav")])
        clips = []
        for w in wavs:
            audio, sr = sf.read(os.path.join(spk_dir, w))
            if sr != SR:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
            clips.append(audio)
        speaker_clips[name] = clips
        total_dur = sum(len(c) / SR for c in clips)
        print(f"  {name}: {len(clips)} clips, {total_dur:.1f}s total")

    # Build conversation order: interleave speakers in a natural pattern
    # Simulate a meeting: speakers take turns, sometimes same speaker continues
    conversation = []
    clip_indices = {name: 0 for name in speaker_clips}
    speaker_names = list(speaker_clips.keys())

    # Create a turn sequence that feels like a real meeting
    random.seed(42)
    turns = []
    for _ in range(50):  # enough turns to fill ~5 min
        # 70% chance of switching speaker, 30% same speaker continues
        if not turns or random.random() < 0.7:
            speaker = random.choice(speaker_names)
        else:
            speaker = turns[-1]
        turns.append(speaker)

    # Build the audio with short pauses between turns
    segments_info = []  # ground truth
    total_audio = np.array([], dtype=np.float64)
    current_time = 0.0

    for speaker in turns:
        idx = clip_indices[speaker]
        clips = speaker_clips[speaker]
        if idx >= len(clips):
            continue  # no more clips for this speaker

        clip = clips[idx]
        clip_indices[speaker] = idx + 1

        # Add a short pause (0.3-0.8s) between turns
        pause = random.uniform(0.3, 0.8)
        silence = np.zeros(int(pause * SR))
        total_audio = np.concatenate([total_audio, silence])
        current_time += pause

        # Add the clip
        start = current_time
        total_audio = np.concatenate([total_audio, clip])
        duration = len(clip) / SR
        current_time += duration

        segments_info.append({
            "start": round(start, 3),
            "end": round(current_time, 3),
            "speaker": speaker,
            "status": SPEAKERS[speaker],
            "duration": round(duration, 1),
        })

    # Write the conversation
    sf.write(OUTPUT_WAV, total_audio, SR)
    total_duration = len(total_audio) / SR

    print(f"\nConversation built: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Saved to: {OUTPUT_WAV}")

    # Print ground truth timeline
    print(f"\n{'Time':<16} {'Speaker':<10} {'Status':<12} {'Duration'}")
    print("-" * 50)
    for seg in segments_info:
        print(f"  [{seg['start']:6.1f}-{seg['end']:6.1f}]  {seg['speaker']:<10} {seg['status']:<12} {seg['duration']}s")

    # Summary
    auth_time = sum(s["duration"] for s in segments_info if s["status"] == "authorized")
    unauth_time = sum(s["duration"] for s in segments_info if s["status"] == "unauthorized")
    print(f"\nAuthorized speech:   {auth_time:.1f}s")
    print(f"Unauthorized speech: {unauth_time:.1f}s")
    print(f"Total speech:        {auth_time + unauth_time:.1f}s")

    # Save ground truth
    import json
    gt_path = os.path.join(DATA_DIR, "long_conversation_ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump({
            "total_duration": round(total_duration, 1),
            "authorized_speakers": [n for n, s in SPEAKERS.items() if s == "authorized"],
            "unauthorized_speakers": [n for n, s in SPEAKERS.items() if s == "unauthorized"],
            "segments": segments_info,
        }, f, indent=2)
    print(f"Ground truth saved to: {gt_path}")

    return total_duration


if __name__ == "__main__":
    dur = build_conversation()
    print(f"\nDone! Now run: python main.py data/long_conversation.wav --output data/long_results.json")
