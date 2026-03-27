# Speaker Diarization & Secure Speaker Recognition

A modular pipeline for speaker diarization and secure speaker recognition in multi-speaker conversations using pretrained models.

## Features

- **Speaker Diarization** - Who spoke when (pyannote.audio 3.1)
- **Speech-to-Text** - Timestamped transcription (WhisperX)
- **Spoof Detection** - Detect AI-generated voices using Wav2Vec2
- **Open-Set Speaker Recognition** - Identify speakers using ECAPA-TDNN embeddings + cosine similarity

## Pipeline Architecture

```
Input Audio
  -> Preprocessing (16kHz mono)
  -> Speaker Diarization (pyannote pretrained pipeline)
  -> ASR (WhisperX)
  -> Alignment (speaker + text fusion)
  -> Mode Switch:
       --mode deepfake : Spoof Detection (Wav2Vec2)  -> REAL / SPOOF
       --mode auth     : Speaker Recognition (ECAPA-TDNN) -> Name / UNKNOWN
  -> Structured JSON Output
```

## Project Structure

```
Speaker-Diarization/
├── main.py                     # Single entry point
├── diarization/
│   └── diarize.py              # pyannote speaker diarization
├── asr/
│   └── transcribe.py           # WhisperX transcription
├── alignment/
│   └── align.py                # Speaker-text alignment
├── embeddings/
│   └── speaker_id.py           # ECAPA-TDNN speaker identification
├── spoof_detection/
│   └── infer.py                # Wav2Vec2 deepfake detection
├── utils/
│   └── audio_utils.py          # Audio preprocessing
├── scripts/
│   ├── enroll_speaker.py       # Enroll a speaker into registry
│   └── setup_speakers.py       # Batch speaker enrollment
├── data/
│   └── speaker_registry.json   # Registered speaker embeddings
├── results/                    # Output JSON files
├── test_pipeline.py            # Module tests
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.10
- GPU recommended (but CPU works)
- HuggingFace account (for pyannote model access)

### Setup

```bash
git clone https://github.com/pvarma-05/Speaker-Diarization.git
cd Speaker-Diarization
pip install -r requirements.txt
```

### HuggingFace Token (Required)

The pipeline uses `pyannote/speaker-diarization-3.1` which is a gated model:

1. Accept terms at [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Set it: `export HF_TOKEN=your_token` or create a `.env` file with `HF_TOKEN=your_token`

## Usage

### Spoof Detection Mode

Detect AI-generated or TTS voices in a conversation:

```bash
python main.py --audio input.wav --mode deepfake
python main.py --audio input.wav --mode deepfake --output results/output.json
```

### Open-Set Speaker Recognition Mode

Identify registered speakers and flag unknown voices:

```bash
python main.py --audio input.wav --mode auth
python main.py --audio input.wav --mode auth --speaker-threshold 0.8
```

### Enroll Speakers (for auth mode)

```bash
python scripts/enroll_speaker.py --name "Alice" --audio path/to/alice.wav
python scripts/enroll_speaker.py --list
python scripts/setup_speakers.py   # Batch enrollment from data/speakers/
```

### Run Tests

```bash
python test_pipeline.py
```

## Output Format

```json
[
  {
    "start": 0.0,
    "end": 3.2,
    "speaker": "Speaker_1",
    "text": "Hello everyone",
    "label": "REAL"
  }
]
```

| Field   | Description |
|---------|-------------|
| start   | Segment start time (seconds) |
| end     | Segment end time (seconds) |
| speaker | Speaker identifier |
| text    | Transcribed text |
| label   | `REAL` / `SPOOF` (deepfake mode) or speaker name / `UNKNOWN` (auth mode) |

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--audio` | required | Path to input audio file |
| `--mode` | `auth` | Pipeline mode: `auth` or `deepfake` |
| `--output` | None | Path to save JSON output |
| `--hf-token` | from .env | HuggingFace API token |
| `--whisper-model` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `--speaker-threshold` | `0.75` | Cosine similarity threshold for speaker recognition |
| `--merge-gap` | `0.5` | Max gap (seconds) to merge adjacent same-speaker segments |
| `--no-auth-filter` | False | Include unknown speakers in output |

## Pretrained Models Used

| Component | Model | Source |
|-----------|-------|--------|
| Diarization | pyannote/speaker-diarization-3.1 | HuggingFace |
| ASR | WhisperX (OpenAI Whisper + alignment) | faster-whisper |
| Speaker Embeddings | ECAPA-TDNN | SpeechBrain (speechbrain/spkrec-ecapa-voxceleb) |
| Spoof Detection | Wav2Vec2 fine-tuned | HuggingFace (mo-thecreator/Deepfake-audio-detection) |

## License

MIT License.
