"""
Download multi-speaker audio from LibriSpeech and prepare test data
for speaker authorization testing.

Creates:
  data/speakers/<name>/enrollment.wav   - for enrolling authorized speakers
  data/speakers/<name>/samples/*.wav    - individual utterances
  data/test_conversation.wav            - mixed conversation from all speakers

Usage: python scripts/setup_test_data.py
"""

import os
import sys
import tarfile
import urllib.request
import shutil
import random
import struct
import wave
import io

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPEAKERS_DIR = os.path.join(DATA_DIR, "speakers")

# LibriSpeech dev-clean URL
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
TAR_PATH = os.path.join(DATA_DIR, "dev-clean.tar.gz")

# We'll pick 4 speakers from dev-clean
# Speaker IDs from LibriSpeech dev-clean (known to exist)
SPEAKER_IDS = ["84", "174", "1272", "1462"]
SPEAKER_NAMES = {
    "84": "Alice",
    "174": "Bob", 
    "1272": "Charlie",
    "1462": "Diana"
}

# Alice and Bob = authorized, Charlie and Diana = unauthorized
AUTHORIZED = ["Alice", "Bob"]
MAX_UTTERANCES_PER_SPEAKER = 5  # Keep it small


def download_librispeech():
    """Download LibriSpeech dev-clean if not already present."""
    if os.path.exists(TAR_PATH):
        print(f"  Already downloaded: {TAR_PATH}")
        return
    
    print(f"  Downloading LibriSpeech dev-clean (~337MB)...")
    print(f"  URL: {LIBRISPEECH_URL}")
    
    def progress_hook(count, block_size, total_size):
        pct = count * block_size * 100 / total_size
        mb = count * block_size / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end="", flush=True)
    
    urllib.request.urlretrieve(LIBRISPEECH_URL, TAR_PATH, reporthook=progress_hook)
    print("\n  Download complete!")


def extract_speakers():
    """Extract only the 4 target speakers from the tar."""
    print("  Extracting speaker audio...")
    
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        print("Error: soundfile and numpy required. Install with: pip install soundfile numpy")
        sys.exit(1)
    
    os.makedirs(SPEAKERS_DIR, exist_ok=True)
    
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        for member in tar.getmembers():
            # LibriSpeech structure: LibriSpeech/dev-clean/SPEAKER/CHAPTER/file.flac
            parts = member.name.replace("\\", "/").split("/")
            if len(parts) < 4:
                continue
            
            speaker_id = parts[2]  # e.g. "84"
            if speaker_id not in SPEAKER_IDS:
                continue
            
            if not member.name.endswith(".flac"):
                continue
            
            speaker_name = SPEAKER_NAMES[speaker_id]
            samples_dir = os.path.join(SPEAKERS_DIR, speaker_name, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            
            # Check if we already have enough
            existing = [f for f in os.listdir(samples_dir) if f.endswith(".wav")]
            if len(existing) >= MAX_UTTERANCES_PER_SPEAKER:
                continue
            
            # Extract and convert FLAC to WAV
            f = tar.extractfile(member)
            if f is None:
                continue
            
            try:
                audio_data, sr = sf.read(io.BytesIO(f.read()))
                
                # Save as WAV
                basename = os.path.splitext(os.path.basename(member.name))[0] + ".wav"
                wav_path = os.path.join(samples_dir, basename)
                sf.write(wav_path, audio_data, sr)
            except Exception as e:
                print(f"    Warning: Failed to process {member.name}: {e}")
                continue
    
    # Print summary
    for sid, name in SPEAKER_NAMES.items():
        samples_dir = os.path.join(SPEAKERS_DIR, name, "samples")
        if os.path.exists(samples_dir):
            count = len([f for f in os.listdir(samples_dir) if f.endswith(".wav")])
            auth = "AUTHORIZED" if name in AUTHORIZED else "unauthorized"
            print(f"    {name} (Speaker {sid}): {count} utterances [{auth}]")


def create_enrollment_audio():
    """Create enrollment WAV for each speaker by concatenating their first 2 samples."""
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        sys.exit(1)
    
    print("  Creating enrollment audio...")
    
    for name in SPEAKER_NAMES.values():
        samples_dir = os.path.join(SPEAKERS_DIR, name, "samples")
        if not os.path.exists(samples_dir):
            continue
        
        wav_files = sorted([f for f in os.listdir(samples_dir) if f.endswith(".wav")])[:2]
        if not wav_files:
            continue
        
        # Concatenate first 2 utterances for enrollment
        chunks = []
        sr = 16000
        for wf in wav_files:
            audio, file_sr = sf.read(os.path.join(samples_dir, wf))
            if file_sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=16000)
            chunks.append(audio)
            chunks.append(np.zeros(int(0.5 * sr)))  # 0.5s gap
        
        enrollment = np.concatenate(chunks)
        enrollment_path = os.path.join(SPEAKERS_DIR, name, "enrollment.wav")
        sf.write(enrollment_path, enrollment, sr)
        print(f"    {name}: enrollment.wav ({len(enrollment)/sr:.1f}s)")


def create_test_conversation():
    """Create a test 'conversation' by interleaving utterances from all speakers."""
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        sys.exit(1)
    
    print("  Building test conversation...")
    
    sr = 16000
    conversation = []
    speaker_order = []  # Track who speaks when
    
    # Collect 1 utterance per speaker (use utterance index 2-4 to avoid overlap w/ enrollment)
    for name in SPEAKER_NAMES.values():
        samples_dir = os.path.join(SPEAKERS_DIR, name, "samples")
        if not os.path.exists(samples_dir):
            continue
        
        wav_files = sorted([f for f in os.listdir(samples_dir) if f.endswith(".wav")])
        # Use utterances 2+ (first 2 used for enrollment)
        test_utterances = wav_files[2:4] if len(wav_files) > 2 else wav_files[-1:]
        
        for wf in test_utterances:
            audio, file_sr = sf.read(os.path.join(samples_dir, wf))
            if file_sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=16000)
            
            start_time = len(np.concatenate(conversation)) / sr if conversation else 0
            conversation.append(audio)
            pause = np.zeros(int(1.0 * sr))  # 1s pause between speakers
            conversation.append(pause)
            end_time = (len(np.concatenate(conversation)) - len(pause)) / sr
            
            auth_status = "AUTHORIZED" if name in AUTHORIZED else "unauthorized"
            speaker_order.append(f"  [{start_time:.1f}s - {end_time:.1f}s] {name} ({auth_status})")
    
    # Save conversation
    full_audio = np.concatenate(conversation)
    conv_path = os.path.join(DATA_DIR, "test_conversation.wav")
    sf.write(conv_path, full_audio, sr)
    
    print(f"  Saved: {conv_path} ({len(full_audio)/sr:.1f}s)")
    print("  Speaker timeline:")
    for entry in speaker_order:
        print(entry)


def enroll_authorized_speakers():
    """Enroll authorized speakers into the registry."""
    sys.path.insert(0, PROJECT_ROOT)
    from embeddings.speaker_id import (
        extract_speaker_embedding, save_speaker_to_registry, load_speaker_encoder
    )
    
    print("  Enrolling authorized speakers...")
    registry_path = os.path.join(DATA_DIR, "speaker_registry.json")
    
    # Clear existing registry
    if os.path.exists(registry_path):
        os.remove(registry_path)
    
    encoder = load_speaker_encoder()
    
    for name in AUTHORIZED:
        enrollment_path = os.path.join(SPEAKERS_DIR, name, "enrollment.wav")
        if not os.path.exists(enrollment_path):
            print(f"    Warning: No enrollment audio for {name}")
            continue
        
        embedding = extract_speaker_embedding(enrollment_path, encoder)
        save_speaker_to_registry(
            speaker_name=name,
            embedding=embedding,
            registry_path=registry_path,
            metadata={"role": "authorized"}
        )
        print(f"    ✓ {name} enrolled")
    
    print(f"  Registry saved: {registry_path}")


def main():
    print("=" * 50)
    print("Speaker Authorization Test Data Setup")
    print("=" * 50)
    
    print("\n1. Downloading LibriSpeech dev-clean...")
    download_librispeech()
    
    print("\n2. Extracting speaker audio...")
    extract_speakers()
    
    print("\n3. Creating enrollment audio...")
    create_enrollment_audio()
    
    print("\n4. Building test conversation...")
    create_test_conversation()
    
    print("\n5. Enrolling authorized speakers...")
    enroll_authorized_speakers()
    
    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    print(f"\nAuthorized:   {', '.join(AUTHORIZED)}")
    print(f"Unauthorized: {', '.join(n for n in SPEAKER_NAMES.values() if n not in AUTHORIZED)}")
    print(f"\nTo test, run:")
    print(f"  python main.py data/test_conversation.wav")
    print(f"\nExpected: Only {' and '.join(AUTHORIZED)}'s speech should appear in the transcript.")


if __name__ == "__main__":
    main()
