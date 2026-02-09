# Secure, Open-Set, Near End-to-End Speaker Diarization and Recognition (S-O-EEND-SDR)

A modular, CLI-based pipeline for speaker-attributed transcription that answers:
- **Who spoke?** (Speaker identification with open-set capability)
- **When did they speak?** (Timestamped segments)
- **What did they say?** (Automatic speech recognition)
- **Is the speaker known or UNKNOWN?** (Open-set speaker identification)
- **Is the voice genuine or spoofed?** (Anti-spoofing detection - placeholder)

## Overview

This project implements a complete speaker diarization and recognition system using pretrained models. It combines:
- **Speaker Diarization**: pyannote.audio for identifying "who spoke when"
- **ASR**: WhisperX for transcription with word-level timestamps
- **Alignment**: Manual Python logic to fuse diarization and ASR outputs
- **Open-Set Speaker ID**: Resemblyzer embeddings with cosine similarity matching
- **Anti-Spoofing**: Placeholder module for future implementation

## Project Structure

```
sdr-project/
├── data/
│   └── sample_audio.wav          # Placeholder for sample audio
│   └── speaker_registry.json     # Known speaker embeddings (auto-created)
├── diarization/
│   └── diarize.py                # Speaker diarization module
├── asr/
│   └── transcribe.py              # ASR transcription module
├── alignment/
│   └── align.py                   # ASR-diarization alignment logic
├── embeddings/
│   └── speaker_id.py              # Open-set speaker identification
├── spoofing/
│   └── spoof_check.py             # Anti-spoofing detection (placeholder)
├── utils/
│   └── audio_utils.py             # Audio preprocessing utilities
├── main.py                        # CLI entry point
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.9 or 3.10
- CUDA-capable GPU (recommended, but CPU will work)
- Git

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd sdr-project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Accept model terms on HuggingFace:**
   - Visit https://huggingface.co/pyannote/speaker-diarization-3.1
   - Accept the model terms
   - If needed, create a HuggingFace token at https://huggingface.co/settings/tokens
   - Use `--hf-token YOUR_TOKEN` when running the pipeline

## Usage

### Basic Usage

```bash
python main.py path/to/audio.wav
```

### Advanced Usage

```bash
# Save results to JSON
python main.py audio.wav --output results.json

# Use larger Whisper model for better accuracy
python main.py audio.wav --whisper-model large

# Adjust speaker identification threshold
python main.py audio.wav --speaker-threshold 0.8

# Provide HuggingFace token
python main.py audio.wav --hf-token YOUR_HF_TOKEN

# Disable spoofing detection
python main.py audio.wav --no-spoof-check

# Combine options
python main.py audio.wav --output results.json --whisper-model large --speaker-threshold 0.75
```

### Command-Line Arguments

- `audio_file`: Path to input audio file (required)
- `--output, -o`: Path to save JSON output file (optional)
- `--hf-token`: HuggingFace token for model access (optional, may be required)
- `--whisper-model`: Whisper model size: tiny, base, small, medium, large (default: base)
- `--speaker-threshold`: Similarity threshold for speaker identification, 0.0-1.0 (default: 0.7)
- `--merge-gap`: Maximum gap in seconds to merge adjacent segments (default: 0.5)
- `--no-spoof-check`: Disable spoofing detection

## Output Format

### CLI Output

The pipeline prints results in the following format:
```
[start-end] SPEAKER_NAME: transcribed text (spoof_flag)
```

Example:
```
[0.00-2.50] Alice: Hello, how are you? (✓)
[2.50-5.00] Bob: I am doing well, thank you. (✓)
[5.00-7.50] UNKNOWN: Can you help me with this? (✓)
```

Where:
- `[start-end]`: Timestamp range in seconds
- `SPEAKER_NAME`: Identified speaker or "UNKNOWN"
- `transcribed text`: ASR transcription
- `(✓)`: Genuine voice (or `(✗)` if spoofed - placeholder always shows ✓)

### JSON Output

The JSON output contains structured data:
```json
{
  "audio_file": "path/to/audio.wav",
  "num_segments": 3,
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, how are you?",
      "speaker_label": "SPEAKER_00",
      "identified_speaker": "Alice",
      "similarity_score": 0.85,
      "spoof_check": {
        "is_genuine": true,
        "confidence": 1.0,
        "method": "placeholder"
      }
    }
  ]
}
```

## Module Usage

Each module can be run independently for testing:

### Diarization
```bash
python diarization/diarize.py audio.wav
```

### ASR Transcription
```bash
python asr/transcribe.py audio.wav
```

### Speaker Identification
```bash
python embeddings/speaker_id.py audio.wav
```

### Spoofing Detection
```bash
python spoofing/spoof_check.py audio.wav
```

## Speaker Registry

The system maintains a registry of known speakers in `data/speaker_registry.json`. To add a known speaker:

```python
from embeddings.speaker_id import extract_speaker_embedding, save_speaker_to_registry

# Extract embedding from reference audio
embedding = extract_speaker_embedding("reference_audio.wav")

# Save to registry
save_speaker_to_registry("Alice", embedding, metadata={"email": "alice@example.com"})
```

The registry is automatically loaded during speaker identification.

## Current Limitations

1. **Anti-Spoofing**: Currently a placeholder that always returns "genuine". Future work needed to integrate pretrained anti-spoofing models.

2. **Speaker Registry**: Manual process to add known speakers. No automatic enrollment from diarization output.

3. **Overlapping Speech**: Basic handling of overlapping speech segments. May not perfectly handle complex overlaps.

4. **Language Support**: Currently configured for English. WhisperX supports multiple languages but alignment model is English-only.

5. **Model Access**: Requires HuggingFace account and token acceptance for pyannote.audio models.

6. **GPU Memory**: Large Whisper models require significant GPU memory. Use smaller models (base/small) if memory is limited.

## Future Work

1. **Anti-Spoofing Integration**:
   - Integrate ASVspoof or AASIST pretrained models
   - Add confidence scores and detailed spoofing analysis

2. **Automatic Speaker Enrollment**:
   - Automatically add speakers to registry after manual verification
   - Support for speaker clustering and naming

3. **Improved Overlap Handling**:
   - Better handling of simultaneous speech
   - Multi-speaker transcription for overlapping segments

4. **Multi-Language Support**:
   - Language detection
   - Language-specific alignment models

5. **Performance Optimization**:
   - Batch processing for multiple files
   - Streaming support for real-time processing
   - Model quantization for faster inference

6. **Enhanced Output Formats**:
   - SRT subtitle file generation
   - RTTM format export
   - WebVTT format support

## Troubleshooting

### "Failed to load diarization pipeline"
- Ensure you've accepted model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
- Provide HuggingFace token with `--hf-token` flag

### "CUDA out of memory"
- Use smaller Whisper model: `--whisper-model tiny` or `--whisper-model base`
- Process shorter audio segments
- Use CPU mode (slower but uses less memory)

### "No module named 'resemblyzer'"
- Install dependencies: `pip install -r requirements.txt`
- If issues persist, resemblyzer may need additional setup

### Poor transcription accuracy
- Use larger Whisper model: `--whisper-model large`
- Ensure audio quality is good (16kHz, mono recommended)
- Check audio file format (WAV, MP3, etc.)

### Speaker identification always returns "UNKNOWN"
- Check that speaker registry exists and contains known speakers
- Lower the similarity threshold: `--speaker-threshold 0.5`
- Ensure audio segments are long enough (>0.5 seconds)

## License

This project is for educational/capstone purposes. Please check individual model licenses:
- pyannote.audio: MIT License
- WhisperX: MIT License
- Resemblyzer: MIT License

## Acknowledgments

- pyannote.audio team for speaker diarization models
- OpenAI for Whisper ASR models
- Resemblyzer for speaker embedding models
- Inspired by SpeakerLM and related research

## Contact

For questions or issues, please refer to the project documentation or contact the project maintainer.
