# Speaker Authorization Pipeline (S-O-EEND-SDR)

A secure, open-set speaker diarization and **authorization** pipeline. It transcribes only **authorized speakers** and filters out unauthorized voices (imposters/uninvited speakers).

## Key Features

- **Speaker Diarization**: Uses `pyannote.audio` to determine "who spoke when".
- **Speaker Authorization**: Verifies speaker identity against a secure registry using `Resemblyzer` embeddings (Cosine Similarity > 0.75).
- **ASR Transcription**: Uses `WhisperX` for accurate, timestamped speech-to-text.
- **Security Filter**: Automatically mutes/removes segments from unauthorized speakers.

## Project Structure

```
sdr-project/
├── data/
│   ├── speaker_registry.json     # Encrypted-like embeddings of authorized users
│   └── speakers/                 # Source audio for enrollment (optional)
├── results/                      # Generated transcripts and logs
├── diarization/                  # Diarization logic (pyannote)
├── asr/                          # Transcription logic (WhisperX)
├── start_here/                   # Entry point scripts
├── scripts/
│   ├── enroll_speaker.py         # CLI to add users to registry
│   ├── benchmark_auth.py         # Test system accuracy
│   └── setup_test_data.py        # Download test samples
├── utilities/                    # Audio processing helpers
├── main.py                       # Main pipeline entry point
└── requirements.txt              # Dependencies
```

## Installation

### 1. Prerequisities
- Python 3.10
- GPU recommended (for faster processing)
- **Hugging Face Account** (Required for pyannote model)

### 2. Setup
```bash
# Clone
git clone https://github.com/pvarma-05/Speaker-Diarization.git
cd Speaker-Diarization

# Install dependencies
pip install -r requirements.txt
```

### 3. **CRITICAL: Hugging Face Token**
This pipeline uses the `pyannote/speaker-diarization-3.1` model, which is gated.
1. Go to [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the terms.
2. Get your Access Token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Set it as an environment variable or pass it via CLI.

### 4. **CRITICAL: Enroll Speakers Locally**
Voiceprint embeddings must be computed on YOUR machine. After cloning, run:
```bash
python scripts/setup_speakers.py
```
This enrolls Alice, Bob, and Charlie from the included audio samples. Only needed once per machine.

## Usage

### 1. Enroll an Authorized Speaker
Before the system can recognize "Alice", you must enroll her voice.
**Requirements:** 3-5 clear audio clips of the speaker.

```bash
python scripts/enroll_speaker.py --name "Alice" --audio path/to/alice.wav
```

### 2. Run the Pipeline
Process a meeting recording. Only "Alice" (and other enrolled users) will be transcribed.

```bash
python main.py meeting_audio.wav --output results/meeting_transcript.json --hf-token YOUR_TOKEN
```

### 3. Benchmarking
Verify system accuracy on your hardware.

```bash
# Run enhanced benchmark (tested on LibriSpeech)
python scripts/benchmark_enhanced.py
```

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--speaker-threshold` | `0.75` | Cosine similarity score (0-1) to accept a speaker. |
| `--min-speakers` | `None` | Hint for diarization. |
| `--max-speakers` | `None` | Hint for diarization. |
| `--no-auth-filter` | `False` | Disable authorization (transcribe everyone). |

## Performance

- **Clean Audio**: ~96.7% Accuracy, 3.8% False Acceptance Rate (FAR).
- **Noisy Audio (20dB)**: ~85% Accuracy, 0% FAR (Secure failure mode).
- **Long Conversation**: 100% Accuracy on 6-minute multi-speaker test.

## License
MIT License.
