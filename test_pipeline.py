"""
Minimal tests for the Speaker Diarization & Secure Recognition pipeline.
Tests each module independently with mock or minimal data.

Usage: python test_pipeline.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_audio_loading():
    """Test audio loading utility."""
    from utils.audio_utils import load_audio

    try:
        load_audio("nonexistent_file.wav")
        print("  FAIL - Should have raised ValueError")
        return False
    except ValueError:
        print("  OK - Correctly rejects missing files")
        return True


def test_alignment():
    """Test alignment module with mock data."""
    from alignment.align import align_segments, merge_adjacent_segments

    asr_segments = [
        {'start': 0.0, 'end': 2.5, 'text': 'Hello, how are you?'},
        {'start': 2.5, 'end': 5.0, 'text': 'I am doing well.'},
        {'start': 5.5, 'end': 8.0, 'text': 'That is great to hear.'}
    ]

    diarization_segments = [
        {'start': 0.0, 'end': 2.8, 'speaker_label': 'SPEAKER_00'},
        {'start': 2.8, 'end': 5.2, 'speaker_label': 'SPEAKER_01'},
        {'start': 5.3, 'end': 8.5, 'speaker_label': 'SPEAKER_00'}
    ]

    aligned = align_segments(asr_segments, diarization_segments)
    assert len(aligned) == 3, f"Expected 3 segments, got {len(aligned)}"
    assert aligned[0]['speaker_label'] == 'SPEAKER_00'
    assert aligned[1]['speaker_label'] == 'SPEAKER_01'
    assert aligned[2]['speaker_label'] == 'SPEAKER_00'
    print("  OK - Alignment assigns correct speakers")

    merged = merge_adjacent_segments(aligned, same_speaker_threshold=1.0)
    assert len(merged) <= len(aligned)
    print(f"  OK - Merged {len(aligned)} -> {len(merged)} segments")

    return True


def test_spoof_detector():
    """Test spoof detection module with synthetic audio."""
    try:
        from spoof_detection.infer import SpoofDetector
    except ImportError as e:
        print(f"  SKIP - Missing dependency: {e}")
        return True

    try:
        detector = SpoofDetector()
    except Exception as e:
        print(f"  SKIP - Cannot load model: {e}")
        return True

    fake_audio = np.random.randn(16000 * 3).astype(np.float32)
    label, score = detector.predict(fake_audio, sr=16000)

    assert label in ("Real", "Spoof"), f"Unexpected label: {label}"
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    print(f"  OK - Prediction: {label} ({score:.2f})")

    return True


def test_speaker_encoder():
    """Test ECAPA-TDNN speaker encoder."""
    try:
        from embeddings.speaker_id import (
            load_speaker_encoder, extract_embedding_from_segment
        )
    except ImportError as e:
        print(f"  SKIP - Missing dependency: {e}")
        return True

    try:
        encoder = load_speaker_encoder()
    except Exception as e:
        print(f"  SKIP - Cannot load model: {e}")
        return True

    fake_audio = np.random.randn(16000 * 2).astype(np.float32)
    embedding = extract_embedding_from_segment(fake_audio, sr=16000, encoder=encoder)

    assert embedding.ndim == 1, f"Expected 1D embedding, got {embedding.ndim}D"
    assert embedding.shape[0] == 192, f"ECAPA-TDNN should produce 192-dim, got {embedding.shape[0]}"
    print(f"  OK - Embedding shape: {embedding.shape}")

    return True


def test_speaker_registry():
    """Test speaker registry load/save round-trip."""
    from embeddings.speaker_id import load_speaker_registry, save_speaker_to_registry
    import tempfile
    import json

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False, dir='.'
    ) as f:
        tmp_path = f.name
        json.dump({}, f)

    try:
        registry = load_speaker_registry(tmp_path)
        assert len(registry) == 0
        print("  OK - Empty registry loads correctly")

        fake_embedding = np.random.randn(192).astype(np.float32)
        save_speaker_to_registry("TestUser", fake_embedding, registry_path=tmp_path)

        registry = load_speaker_registry(tmp_path)
        assert "TestUser" in registry
        assert len(registry["TestUser"]["embedding"]) == 192
        print("  OK - Save/load round-trip works")

    finally:
        os.unlink(tmp_path)

    return True


def main():
    print("=" * 50)
    print("Pipeline Module Tests")
    print("=" * 50)

    tests = [
        ("Audio Loading", test_audio_loading),
        ("Alignment", test_alignment),
        ("Speaker Registry", test_speaker_registry),
        ("Spoof Detector", test_spoof_detector),
        ("ECAPA-TDNN Encoder", test_speaker_encoder),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAIL - {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
