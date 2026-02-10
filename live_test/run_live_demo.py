"""
Run the live demo end-to-end.

1. Enrolls YOUR voice into an isolated registry (live_test/live_registry.json)
2. Runs the pipeline on live_conversation.wav
3. Shows that only YOUR segments are transcribed

Usage: python live_test/run_live_demo.py
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

LIVE_DIR = os.path.dirname(os.path.abspath(__file__))
MY_VOICE = os.path.join(LIVE_DIR, "my_voice.wav")
LIVE_CONVERSATION = os.path.join(LIVE_DIR, "live_conversation.wav")
LIVE_REGISTRY = os.path.join(LIVE_DIR, "live_registry.json")
LIVE_RESULTS = os.path.join(LIVE_DIR, "live_results.json")


def run_demo():
    """Run the complete live demo pipeline."""
    print("=" * 60)
    print("  LIVE DEMO - Speaker Authorization Pipeline")
    print("=" * 60)
    print()
    
    # Validate files
    if not os.path.exists(MY_VOICE):
        print("  ERROR: Record your voice first!")
        print("  Run: python live_test/record_voice.py")
        sys.exit(1)
    
    if not os.path.exists(LIVE_CONVERSATION):
        print("  ERROR: Build the live conversation first!")
        print("  Run: python live_test/build_live_conversation.py")
        sys.exit(1)
    
    # ── Step 1: Enroll the user ──────────────────────────────────
    print("  STEP 1: Enrolling your voice...")
    print()
    
    # Clean previous registry if exists
    if os.path.exists(LIVE_REGISTRY):
        os.remove(LIVE_REGISTRY)
    
    from embeddings.speaker_id import (
        extract_speaker_embedding,
        save_speaker_to_registry,
        load_speaker_encoder
    )
    
    encoder = load_speaker_encoder()
    embedding = extract_speaker_embedding(MY_VOICE, encoder)
    
    if embedding is None:
        print("  ERROR: Failed to extract voiceprint. Check audio quality.")
        sys.exit(1)
    
    save_speaker_to_registry(
        speaker_name="DEMO_USER",
        embedding=embedding,
        registry_path=LIVE_REGISTRY,
        metadata={"source": "live_recording"}
    )
    print(f"  ✓ Enrolled as 'DEMO_USER' in {LIVE_REGISTRY}")
    print()
    
    # ── Step 2: Run the pipeline ─────────────────────────────────
    print("  STEP 2: Running pipeline on live conversation...")
    print("  (Only YOUR segments should appear in the output)")
    print()
    
    from main import run_pipeline
    
    result = run_pipeline(
        audio_file=LIVE_CONVERSATION,
        output_json=LIVE_RESULTS,
        whisper_model="base",
        speaker_threshold=0.75,
        enable_auth_filter=True,
        registry_path=LIVE_REGISTRY
    )
    
    # ── Step 3: Show results ─────────────────────────────────────
    print()
    print("=" * 60)
    print("  DEMO RESULTS")
    print("=" * 60)
    print()
    
    segments = result.get("segments", [])
    total_input = result.get("total_segments_before_filter", "?")
    
    print(f"  Total segments in audio  : {total_input}")
    print(f"  Authorized (your voice)  : {len(segments)}")
    print(f"  Filtered out (others)    : {int(total_input) - len(segments) if isinstance(total_input, int) else '?'}")
    print()
    
    if segments:
        print("  ── Your Transcribed Segments ──")
        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            speaker = seg.get("identified_speaker", "?")
            text = seg.get("text", "")
            sim = seg.get("similarity_score", 0)
            print(f"  [{start:.1f}-{end:.1f}] {speaker} (sim={sim:.2f}): {text}")
        print()
    else:
        print("  ⚠ No authorized segments found!")
        print("  This could mean your voice wasn't matched.")
        print("  Try recording again with clearer audio.")
        print()
    
    print(f"  Full results saved to: {LIVE_RESULTS}")
    print()
    print("  Demo complete! 🎉")


if __name__ == "__main__":
    run_demo()
