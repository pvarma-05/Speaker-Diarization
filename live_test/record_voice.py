"""
Record your voice for the live demo.

Records 10-15 seconds of audio from your microphone,
saves it as live_test/my_voice.wav.

Usage: python live_test/record_voice.py
"""

import os
import sys
import time
import numpy as np

# Check for sounddevice
try:
    import sounddevice as sd
except ImportError:
    print("ERROR: 'sounddevice' is not installed.")
    print("Run:  pip install sounddevice")
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("ERROR: 'soundfile' is not installed.")
    print("Run:  pip install soundfile")
    sys.exit(1)

SAMPLE_RATE = 16000
DURATION = 15  # seconds
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "my_voice.wav")


def record_voice():
    """Record audio from the default microphone."""
    print("=" * 50)
    print("  VOICE RECORDER - Live Demo Setup")
    print("=" * 50)
    print()
    print(f"  Duration : {DURATION} seconds")
    print(f"  Output   : {OUTPUT_FILE}")
    print()
    print("  TIP: Speak clearly and naturally.")
    print("  Say something like:")
    print('    "Hello, my name is [Your Name].')
    print('     I am testing the speaker authorization system.')
    print('     This system should only transcribe my voice')
    print('     and filter out everyone else."')
    print()
    
    input("  Press ENTER when ready to record... ")
    
    print()
    print("  🔴 RECORDING... Speak now!")
    print()
    
    # Record
    audio = sd.rec(int(DURATION * SAMPLE_RATE), 
                   samplerate=SAMPLE_RATE, 
                   channels=1, 
                   dtype='float32')
    
    # Show countdown
    for i in range(DURATION, 0, -1):
        sys.stdout.write(f"\r  Time remaining: {i:2d}s ")
        sys.stdout.flush()
        time.sleep(1)
    
    sd.wait()  # Wait until recording is finished
    
    print("\r  ✓ Recording complete!          ")
    print()
    
    # Normalize audio
    audio = audio.flatten()
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    
    # Check if audio has content (not silence)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.01:
        print("  ⚠ WARNING: Audio seems very quiet. Check your microphone.")
    
    # Save
    sf.write(OUTPUT_FILE, audio, SAMPLE_RATE)
    print(f"  ✓ Saved to: {OUTPUT_FILE}")
    print(f"  ✓ Duration: {len(audio)/SAMPLE_RATE:.1f}s")
    print(f"  ✓ RMS level: {rms:.4f}")
    print()
    
    return OUTPUT_FILE


if __name__ == "__main__":
    record_voice()
