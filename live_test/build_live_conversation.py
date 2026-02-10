"""
Build a live demo conversation by splicing YOUR voice into the existing conversation.

Takes the pre-built long_conversation.wav (5 speakers) and inserts your
recorded voice at 3 points (start, middle, end) to create a mixed
conversation where YOU are the only authorized speaker.

Usage: python live_test/build_live_conversation.py
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import soundfile as sf
except ImportError:
    print("ERROR: 'soundfile' not installed. Run: pip install soundfile")
    sys.exit(1)

LIVE_DIR = os.path.dirname(os.path.abspath(__file__))
MY_VOICE = os.path.join(LIVE_DIR, "my_voice.wav")
SOURCE_CONVERSATION = os.path.join(PROJECT_ROOT, "data", "long_conversation.wav")
OUTPUT_CONVERSATION = os.path.join(LIVE_DIR, "live_conversation.wav")
SAMPLE_RATE = 16000


def add_silence(duration_sec):
    """Generate silence of given duration."""
    return np.zeros(int(duration_sec * SAMPLE_RATE), dtype=np.float32)


def build_live_conversation():
    """Splice user's voice into the existing conversation."""
    print("=" * 50)
    print("  BUILD LIVE DEMO CONVERSATION")
    print("=" * 50)
    print()
    
    # Check files exist
    if not os.path.exists(MY_VOICE):
        print(f"  ERROR: Your voice recording not found!")
        print(f"  Run 'python live_test/record_voice.py' first.")
        sys.exit(1)
    
    if not os.path.exists(SOURCE_CONVERSATION):
        print(f"  ERROR: Source conversation not found: {SOURCE_CONVERSATION}")
        print(f"  Run 'python scripts/build_long_conversation.py' first.")
        sys.exit(1)
    
    # Load audio
    print(f"  Loading your voice: {MY_VOICE}")
    my_audio, sr1 = sf.read(MY_VOICE, dtype='float32')
    if sr1 != SAMPLE_RATE:
        print(f"  WARNING: Resampling from {sr1} to {SAMPLE_RATE}")
    
    print(f"  Loading source conversation: {SOURCE_CONVERSATION}")
    conv_audio, sr2 = sf.read(SOURCE_CONVERSATION, dtype='float32')
    if sr2 != SAMPLE_RATE:
        print(f"  WARNING: Resampling from {sr2} to {SAMPLE_RATE}")
    
    my_dur = len(my_audio) / SAMPLE_RATE
    conv_dur = len(conv_audio) / SAMPLE_RATE
    print(f"  Your voice: {my_dur:.1f}s")
    print(f"  Conversation: {conv_dur:.1f}s")
    print()
    
    # Split user's voice into 3 chunks
    chunk_size = len(my_audio) // 3
    chunk1 = my_audio[:chunk_size]
    chunk2 = my_audio[chunk_size:2*chunk_size]
    chunk3 = my_audio[2*chunk_size:]
    
    # Split conversation into 3 parts
    conv_third = len(conv_audio) // 3
    part1 = conv_audio[:conv_third]
    part2 = conv_audio[conv_third:2*conv_third]
    part3 = conv_audio[2*conv_third:]
    
    silence = add_silence(0.5)  # 0.5s gap between speakers
    
    # Build: [YOUR_VOICE] [gap] [conv_part1] [gap] [YOUR_VOICE] [gap] [conv_part2] [gap] [YOUR_VOICE] [gap] [conv_part3]
    print("  Splicing conversation...")
    live_audio = np.concatenate([
        chunk1, silence,
        part1, silence,
        chunk2, silence,
        part2, silence,
        chunk3, silence,
        part3
    ])
    
    total_dur = len(live_audio) / SAMPLE_RATE
    print(f"  Total duration: {total_dur:.1f}s ({total_dur/60:.1f} min)")
    
    # Save
    sf.write(OUTPUT_CONVERSATION, live_audio, SAMPLE_RATE)
    print(f"  ✓ Saved: {OUTPUT_CONVERSATION}")
    print()
    print("  Your voice appears at:")
    t = 0.0
    print(f"    1. [{t:.1f}s - {t + len(chunk1)/SAMPLE_RATE:.1f}s]")
    t += len(chunk1)/SAMPLE_RATE + 0.5 + len(part1)/SAMPLE_RATE + 0.5
    print(f"    2. [{t:.1f}s - {t + len(chunk2)/SAMPLE_RATE:.1f}s]")
    t += len(chunk2)/SAMPLE_RATE + 0.5 + len(part2)/SAMPLE_RATE + 0.5
    print(f"    3. [{t:.1f}s - {t + len(chunk3)/SAMPLE_RATE:.1f}s]")
    print()
    
    return OUTPUT_CONVERSATION


if __name__ == "__main__":
    build_live_conversation()
