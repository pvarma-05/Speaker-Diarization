"""
First-time setup: Enroll all authorized speakers from their audio samples.

Run this ONCE after cloning the repo to build the speaker registry
on YOUR machine. This ensures the voiceprint embeddings are computed
locally, which is required for accurate speaker matching.

Usage: python scripts/setup_speakers.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from embeddings.speaker_id import (
    extract_speaker_embedding,
    save_speaker_to_registry,
    load_speaker_encoder
)

SPEAKERS_DIR = os.path.join(PROJECT_ROOT, "data", "speakers")
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "speaker_registry.json")

# Only these speakers are authorized; others (e.g. Diana) are kept unauthorized for demo purposes
AUTHORIZED_SPEAKERS = {"Alice", "Bob", "Charlie"}


def setup_speakers():
    """Enroll all speakers found in data/speakers/*/enrollment.wav"""
    print("=" * 50)
    print("  SPEAKER ENROLLMENT SETUP")
    print("=" * 50)
    print()
    
    if not os.path.exists(SPEAKERS_DIR):
        print(f"  ERROR: Speakers directory not found: {SPEAKERS_DIR}")
        print(f"  Run scripts/setup_test_data.py first to download audio.")
        sys.exit(1)
    
    # Find authorized speaker directories only
    speaker_dirs = [d for d in os.listdir(SPEAKERS_DIR) 
                    if os.path.isdir(os.path.join(SPEAKERS_DIR, d))
                    and d in AUTHORIZED_SPEAKERS]
    
    skipped = [d for d in os.listdir(SPEAKERS_DIR)
               if os.path.isdir(os.path.join(SPEAKERS_DIR, d))
               and d not in AUTHORIZED_SPEAKERS]
    
    if not speaker_dirs:
        print("  No speaker directories found.")
        sys.exit(1)
    
    print(f"  Found {len(speaker_dirs)} authorized speakers: {', '.join(sorted(speaker_dirs))}")
    if skipped:
        print(f"  Skipping {len(skipped)} unauthorized: {', '.join(sorted(skipped))}")
    print()
    
    # Clear existing registry to start fresh
    if os.path.exists(REGISTRY_PATH):
        os.remove(REGISTRY_PATH)
        print("  Cleared existing registry.")
    
    # Load encoder once (reuse for all speakers)
    print("  Loading speaker encoder model...")
    encoder = load_speaker_encoder()
    print("  ✓ Encoder loaded")
    print()
    
    enrolled = 0
    for speaker_name in sorted(speaker_dirs):
        speaker_dir = os.path.join(SPEAKERS_DIR, speaker_name)
        
        # Look for enrollment.wav first, then any .wav file
        enrollment_file = os.path.join(speaker_dir, "enrollment.wav")
        if not os.path.exists(enrollment_file):
            # Try samples directory
            samples_dir = os.path.join(speaker_dir, "samples")
            if os.path.exists(samples_dir):
                wav_files = [f for f in os.listdir(samples_dir) if f.endswith('.wav')]
                if wav_files:
                    enrollment_file = os.path.join(samples_dir, wav_files[0])
                else:
                    print(f"  ✗ {speaker_name}: No WAV files found, skipping")
                    continue
            else:
                print(f"  ✗ {speaker_name}: No enrollment.wav or samples/ found, skipping")
                continue
        
        print(f"  Enrolling {speaker_name}...", end=" ")
        
        try:
            embedding = extract_speaker_embedding(enrollment_file, encoder)
            if embedding is None:
                print("FAILED (no embedding)")
                continue
            
            save_speaker_to_registry(
                speaker_name=speaker_name,
                embedding=embedding,
                registry_path=REGISTRY_PATH,
                metadata={"role": "authorized", "source": os.path.basename(enrollment_file)}
            )
            enrolled += 1
            print(f"✓ (from {os.path.basename(enrollment_file)})")
        except Exception as e:
            print(f"FAILED ({str(e)})")
    
    print()
    print(f"  ✓ Enrolled {enrolled}/{len(speaker_dirs)} speakers")
    print(f"  ✓ Registry saved to: {REGISTRY_PATH}")
    print()
    print("  You can now run the pipeline:")
    print("    python main.py data/long_conversation.wav --output results/output.json")


if __name__ == "__main__":
    setup_speakers()
