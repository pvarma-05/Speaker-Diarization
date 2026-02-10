"""
Enroll a speaker into the speaker registry for authorization.

Usage:
    python scripts/enroll_speaker.py --name "Alice" --audio path/to/alice_voice.wav
    python scripts/enroll_speaker.py --name "Bob" --audio path/to/bob_voice.wav
    python scripts/enroll_speaker.py --list   # List all enrolled speakers
"""

import os
import sys
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from embeddings.speaker_id import (
    extract_speaker_embedding,
    save_speaker_to_registry,
    load_speaker_registry,
    load_speaker_encoder
)


def enroll_speaker(name: str, audio_file: str, registry_path: str = None):
    """Enroll a speaker by extracting their voiceprint from an audio sample."""
    if registry_path is None:
        registry_path = os.path.join(PROJECT_ROOT, "data", "speaker_registry.json")
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    print(f"Enrolling speaker: {name}")
    print(f"Audio sample: {audio_file}")
    
    # Extract embedding
    print("Extracting voiceprint...")
    encoder = load_speaker_encoder()
    embedding = extract_speaker_embedding(audio_file, encoder)
    
    if embedding is None:
        print("Error: Failed to extract voiceprint. Check audio quality.")
        sys.exit(1)
    
    # Save to registry
    save_speaker_to_registry(
        speaker_name=name,
        embedding=embedding,
        registry_path=registry_path,
        metadata={"source_audio": os.path.basename(audio_file)}
    )
    
    print(f"✓ Speaker '{name}' enrolled successfully!")
    print(f"  Registry: {registry_path}")
    print(f"  Embedding shape: {embedding.shape}")


def list_speakers(registry_path: str = None):
    """List all enrolled speakers."""
    if registry_path is None:
        registry_path = os.path.join(PROJECT_ROOT, "data", "speaker_registry.json")
    
    registry = load_speaker_registry(registry_path)
    
    if not registry:
        print("No speakers enrolled yet.")
        print(f"Use: python scripts/enroll_speaker.py --name \"Name\" --audio sample.wav")
        return
    
    print(f"Enrolled speakers ({len(registry)}):")
    print("-" * 40)
    for name, data in registry.items():
        meta = data.get("metadata", {})
        source = meta.get("source_audio", "unknown")
        print(f"  • {name}  (source: {source})")


def main():
    parser = argparse.ArgumentParser(
        description="Enroll speakers for authorization in the S-O-EEND-SDR pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/enroll_speaker.py --name "Alice" --audio data/alice.wav
  python scripts/enroll_speaker.py --list
        """
    )
    
    parser.add_argument('--name', '-n', type=str,
                       help='Speaker name to enroll')
    parser.add_argument('--audio', '-a', type=str,
                       help='Path to audio sample of the speaker (WAV, 5-30s recommended)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all enrolled speakers')
    parser.add_argument('--registry', type=str, default=None,
                       help='Path to speaker registry JSON file')
    
    args = parser.parse_args()
    
    if args.list:
        list_speakers(args.registry)
    elif args.name and args.audio:
        enroll_speaker(args.name, args.audio, args.registry)
    else:
        parser.print_help()
        print("\nError: Provide --name and --audio to enroll, or --list to view speakers.")
        sys.exit(1)


if __name__ == "__main__":
    main()
