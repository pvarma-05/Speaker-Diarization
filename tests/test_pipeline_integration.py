"""
Integration tests for the Speaker Authorization pipeline.

Tests cover:
- Format output function with authorization status
- Pipeline result structure
- Speaker identification accuracy
"""

import os
import sys
import json
import unittest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

SAMPLE_WAV = os.path.join(PROJECT_ROOT, "data", "sample.wav")
HAS_SAMPLE = os.path.exists(SAMPLE_WAV)


class TestFormatFinalOutput(unittest.TestCase):
    """Tests for the format_final_output function."""

    def test_authorized_segment(self):
        """Authorized segment should show speaker name and similarity."""
        from main import format_final_output
        segments = [{
            'start': 0.0, 'end': 5.0,
            'identified_speaker': 'Alice',
            'text': 'Hello world',
            'similarity_score': 0.85,
        }]
        output = format_final_output(segments)
        self.assertIn('Alice', output)
        self.assertIn('Hello world', output)
        self.assertIn('AUTHORIZED', output)

    def test_unknown_segment(self):
        """UNKNOWN segment should show unauthorized status."""
        from main import format_final_output
        segments = [{
            'start': 1.0, 'end': 3.0,
            'identified_speaker': 'UNKNOWN',
            'text': 'Test text',
            'similarity_score': 0.45,
        }]
        output = format_final_output(segments)
        self.assertIn('UNAUTHORIZED', output)

    def test_empty_segments(self):
        """Empty segment list should produce empty output."""
        from main import format_final_output
        output = format_final_output([])
        self.assertEqual(output, '')

    def test_multiple_segments(self):
        """Multiple segments should each appear on separate lines."""
        from main import format_final_output
        segments = [
            {'start': 0.0, 'end': 5.0, 'identified_speaker': 'Alice',
             'text': 'First', 'similarity_score': 0.9},
            {'start': 5.0, 'end': 10.0, 'identified_speaker': 'Bob',
             'text': 'Second', 'similarity_score': 0.8},
        ]
        output = format_final_output(segments)
        lines = output.strip().split('\n')
        self.assertEqual(len(lines), 2)


class TestSpeakerIdentification(unittest.TestCase):
    """Tests for speaker embedding and identification functions."""

    def test_cosine_similarity(self):
        """Cosine similarity should return correct values."""
        import numpy as np
        from embeddings.speaker_id import cosine_similarity
        
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0, places=5)
        
        c = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, c), 0.0, places=5)

    def test_load_empty_registry(self):
        """Loading non-existent registry should return empty dict."""
        from embeddings.speaker_id import load_speaker_registry
        registry = load_speaker_registry("nonexistent_path.json")
        self.assertEqual(registry, {})

    def test_identify_unknown_speaker(self):
        """Speaker not in registry should be identified as UNKNOWN."""
        import numpy as np
        from embeddings.speaker_id import identify_speaker
        
        emb = np.random.randn(256)
        name, sim = identify_speaker(emb, registry={}, threshold=0.75)
        self.assertEqual(name, "UNKNOWN")


@unittest.skipUnless(HAS_SAMPLE, f"Sample audio not found at {SAMPLE_WAV}")
class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the full pipeline (requires sample.wav + HF token)."""

    @classmethod
    def setUpClass(cls):
        """Load HF token once for all tests."""
        from main import load_hf_token
        cls.token = load_hf_token()
        if not cls.token:
            cls.token = os.environ.get('HF_TOKEN')

    @unittest.skipUnless(os.environ.get('HF_TOKEN') or 
                         os.path.exists(os.path.join(PROJECT_ROOT, '.env')),
                         "HF_TOKEN not available")
    def test_full_pipeline_with_auth_filter(self):
        """Full pipeline should run with authorization filter enabled."""
        from main import run_pipeline
        result = run_pipeline(
            audio_file=SAMPLE_WAV,
            hf_token=self.token,
            whisper_model="base",
            enable_auth_filter=True
        )
        self.assertIn('segments', result)

    @unittest.skipUnless(os.environ.get('HF_TOKEN') or 
                         os.path.exists(os.path.join(PROJECT_ROOT, '.env')),
                         "HF_TOKEN not available")
    def test_full_pipeline_without_auth_filter(self):
        """Pipeline should work with authorization filter disabled."""
        from main import run_pipeline
        result = run_pipeline(
            audio_file=SAMPLE_WAV,
            hf_token=self.token,
            whisper_model="base",
            enable_auth_filter=False
        )
        self.assertIn('segments', result)
        self.assertGreater(len(result['segments']), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
