# Speaker Diarization with Spoof Detection and Open-Set Recognition

This project provides a **unified, cascaded pipeline** that processes a single audio file end-to-end: it estimates who spoke when (diarization), transcribes speech (ASR), assigns words to speakers (alignment), and then, **for every segment**, runs **spoof detection** (authenticity) and **open-set speaker recognition** (identity). Results are written as structured JSON suitable for auditing, downstream analytics, or integration into larger systems.

The design keeps **identity** and **authenticity** separate: a segment may be labeled synthetic while still resolving to an enrolled speaker name or a generic `Speaker_N` label, depending on the embedding comparison with the registry.

---

## Features

- **Speaker diarization** using a pretrained pyannote pipeline (`pyannote/speaker-diarization-3.1`) to segment audio by speaker identity clusters over time.
- **Automatic speech recognition** using WhisperX (OpenAI Whisper via faster-whisper) with forced alignment for timestamped segments.
- **Speaker–text alignment** that maps each ASR segment to the best-matching diarization region by temporal overlap, with optional merging of adjacent same-speaker segments.
- **Unified downstream analysis**: every aligned segment is processed by both the spoof classifier and the ECAPA-TDNN speaker encoder.
- **Identity vs authenticity separation**:
  - **Speaker** and **type** reflect open-set recognition against a JSON registry (`AUTHORIZED` vs `UNAUTHORIZED`, name vs `Speaker_1`, `Speaker_2`, …).
  - **Label** reflects spoof detection only (`REAL` vs `SPOOF`).
- **Segment-level processing** with per-segment transcript, timestamps, and labels.
- **Structured JSON output** including run metadata (`generated_at`), effective configuration (`config`), aggregate **summary** statistics, and a full **segments** array (no removal of “unknown” segments).

---

## System Architecture

Processing follows a single linear pipeline:

1. **Load and resample** the input to 16 kHz mono (librosa).
2. **Diarize** with pyannote to obtain `(start, end, speaker_label)` turns.
3. **Transcribe** with WhisperX; align words to audio for accurate segment times (English alignment model).
4. **Align** ASR segments to diarization by overlap; optionally **merge** adjacent segments from the same diarization speaker if the gap is below a configurable threshold.
5. For **each** merged segment:
   - Crop the waveform and run **spoof detection** (Hugging Face Wav2Vec2 classifier).
   - If the crop is long enough (approximately **0.5 s** or more), extract an **ECAPA-TDNN** embedding and compare it to the **speaker registry** (cosine similarity vs threshold).
   - Emit `speaker`, `type`, `label`, and `text` for that segment.
6. **Write JSON** to `results/<input_basename>.json` under the project root.

```text
Audio (file)
  → Preprocess (16 kHz mono)
  → Diarization (pyannote)
  → ASR + alignment (WhisperX)
  → Fuse speakers + text (alignment + merge)
  → Per segment: Spoof (Wav2Vec2) + Embedding vs registry (ECAPA-TDNN)
  → JSON output
```

---

## Installation and Setup

### Python version

The repository targets **Python 3.10** (see comments in `requirements.txt`). Other 3.x versions may work but are not guaranteed with the pinned stack.

### Dependencies

Install from the project root:

```bash
pip install -r requirements.txt
```

Key stack elements include: **PyTorch** / **torchaudio**, **pyannote.audio** (diarization), **WhisperX** / **faster-whisper** (ASR), **SpeechBrain** (ECAPA-TDNN), **transformers** / **Hugging Face Hub** (spoof model and downloads), **librosa** / **soundfile** (audio I/O), and **numpy** / **scipy**.

### Hugging Face access

- **Pyannote** models are gated. Accept the model conditions on the model card (e.g. `pyannote/speaker-diarization-3.1`), then create an access token at [Hugging Face settings](https://huggingface.co/settings/tokens).
- Set the token before running:

  ```bash
  set HF_TOKEN=your_token_here
  ```

  On Linux or macOS, use `export HF_TOKEN=...`. Alternatively, place `HF_TOKEN=...` in a `.env` file in the project root; `main.py` reads it when the environment variable is unset.

### Model downloads

On first use, components download weights automatically:

- **Pyannote** diarization (via `huggingface_hub` and your token).
- **WhisperX** / faster-whisper ASR weights for the selected `--whisper-model`.
- **SpeechBrain** ECAPA-TDNN encoder into `pretrained_models/ecapa-tdnn/` (see `embeddings/speaker_id.py`).
- **Spoof detection** Wav2Vec2 weights from `mo-thecreator/Deepfake-audio-detection` (see `spoof_detection/infer.py`).

Ensure sufficient disk space and network access for these caches.

---

## Speaker Enrollment

### What the speaker registry is

The **speaker registry** is a JSON file (default: `data/speaker_registry.json`) mapping **logical names** to **fixed-dimensional embeddings** (ECAPA-TDNN vectors) plus optional metadata (e.g. source filename). At inference time, each segment embedding is compared to these entries; if the best cosine similarity meets the threshold, the segment is **AUTHORIZED** and labeled with the enrolled name.

### Why enrollment matters

Open-set recognition requires **known** voiceprints. Without enrollment (or with an empty registry), every segment is treated as **UNAUTHORIZED** and labeled `Speaker_1`, `Speaker_2`, … according to diarization cluster identity, while spoof detection still runs independently.

### How to enroll a speaker

From the project root, run:

```bash
python scripts/enroll_speaker.py --name "Aaron" --audio data/aaron.wav
```

List enrolled speakers:

```bash
python scripts/enroll_speaker.py --list
```

Optional: `--registry path/to/custom_registry.json` to use a non-default file (must match `--registry` when running `main.py`).

### Best practices

- Use **clean, single-speaker** clips with minimal background noise and stable volume.
- Prefer **several seconds** of speech (the enrollment script suggests roughly 5–30 seconds as a practical range).
- **Re-enrolling** the same `--name` **averages** the new embedding with the existing one in the registry, which can approximate combining multiple samples without a separate batch tool.
- Store the registry under version control only if policy allows; embeddings are sensitive biometric information.

---

## How to Run

```bash
python main.py path/to/audio.wav
```

The pipeline is **always unified**: diarization, ASR, alignment, spoof detection, and speaker recognition run together.

**Output file location:** JSON is written to **`results/<basename>.json`**, where `<basename>` is the input filename without extension (e.g. `audio.wav` → `results/audio.json`). The directory `results/` is created if needed.

**Optional arguments:**

| Argument | Description |
|----------|-------------|
| `--hf-token` | Hugging Face token (overrides `HF_TOKEN` / `.env`) |
| `--whisper-model` | `tiny`, `base`, `small`, `medium`, or `large` (default: `base`) |
| `--speaker-threshold` | Cosine similarity threshold for registry match (default: `0.75`) |
| `--merge-gap` | Maximum gap in seconds to merge adjacent same-speaker segments (default: `0.5`) |
| `--registry` | Path to `speaker_registry.json` (default: `data/speaker_registry.json`) |

---

## Output Format

The top-level JSON object has the following fields:

| Field | Description |
|-------|-------------|
| `audio_file` | Basename of the input audio file |
| `generated_at` | Local timestamp string when the run finished (`YYYY-MM-DD HH:MM:SS`) |
| `config` | Effective parameters for this run: `speaker_threshold`, `merge_gap`, `whisper_model` |
| `summary` | Counts: `total_speakers`, `real_segments`, `spoof_segments`, `authorized_segments`, `unauthorized_segments` |
| `segments` | Array of per-segment records |

Each element of `segments` contains:

| Field | Type | Description |
|-------|------|-------------|
| `start` | float | Segment start (seconds) |
| `end` | float | Segment end (seconds) |
| `speaker` | string | Enrolled name if matched, else `Speaker_1`, `Speaker_2`, … |
| `type` | string | `AUTHORIZED` or `UNAUTHORIZED` |
| `label` | string | `REAL` or `SPOOF` (from spoof detector) |
| `text` | string | Transcribed text for the segment |

Example (illustrative):

```json
{
  "audio_file": "meeting.wav",
  "generated_at": "2026-03-28 14:30:00",
  "config": {
    "speaker_threshold": 0.75,
    "merge_gap": 0.5,
    "whisper_model": "base"
  },
  "summary": {
    "total_speakers": 2,
    "real_segments": 12,
    "spoof_segments": 1,
    "authorized_segments": 5,
    "unauthorized_segments": 8
  },
  "segments": [
    {
      "start": 0.0,
      "end": 3.45,
      "speaker": "Aaron",
      "type": "AUTHORIZED",
      "label": "REAL",
      "text": "Let's begin the review."
    },
    {
      "start": 3.45,
      "end": 7.1,
      "speaker": "Speaker_1",
      "type": "UNAUTHORIZED",
      "label": "SPOOF",
      "text": "Thank you for joining."
    }
  ]
}
```

**Summary definitions:**

- `total_speakers`: number of distinct `speaker` strings across all segments.
- `real_segments` / `spoof_segments`: counts of `label == "REAL"` / `"SPOOF"`.
- `authorized_segments` / `unauthorized_segments`: counts of `type == "AUTHORIZED"` / `"UNAUTHORIZED"`.

---

## Explanation of Labels

| Concept | Field | Meaning |
|---------|--------|---------|
| **Who (identity)** | `speaker` | Resolved display name: an enrolled name when the segment embedding matches the registry above the threshold; otherwise a stable `Speaker_N` per diarization cluster for unmatched segments. |
| **Authorization** | `type` | `AUTHORIZED` if matched to the registry; `UNAUTHORIZED` if not (including empty registry or below-threshold similarity). |
| **Authenticity** | `label` | `REAL` or `SPOOF` from the Wav2Vec2 spoof classifier only. Independent of `speaker` and `type`. |

Important: **`SPOOF` is not a substitute for speaker identity.** A deepfake segment can still show an enrolled name if the embedding comparison fires; reviewers should interpret `label` and `type` together.

---

## Technologies Used

| Technology | Role in this project |
|------------|----------------------|
| **Pyannote.audio** | Speaker diarization pipeline |
| **WhisperX** (OpenAI Whisper + alignment) | ASR and timestamped segments |
| **SpeechBrain** | ECAPA-TDNN speaker embeddings |
| **Transformers / Wav2Vec2** | Spoof vs real audio classification |
| **PyTorch** | Deep learning runtime for diarization, ASR, embeddings, and spoof model |
| **Hugging Face Hub** | Model hosting and authentication for pyannote and spoof weights |

Supporting libraries include **librosa**, **faster-whisper**, **numpy**, and **scipy**.

---

## Limitations

- **Short segments** (under roughly half a second of audio) skip reliable embedding extraction; those segments default to **UNAUTHORIZED** with a `Speaker_N` label derived from diarization, while spoof detection may still run on available samples.
- **Threshold sensitivity**: the `--speaker-threshold` trades off false accepts and false rejects; optimal values depend on microphone, room, language, and enrollment quality.
- **Spoof detection** is model-dependent; performance on unseen synthesis methods, heavy compression, or channel effects may degrade.
- **ASR alignment** in the current code path uses an **English** alignment model; other languages may need code changes in `asr/transcribe.py`.
- **CPU vs GPU**: all components can run on CPU, but latency and throughput are significantly better on GPU where supported.

---

## Future Improvements

- **Threshold calibration** using development data (e.g. equal error rate targets) instead of a single manual cosine cutoff.
- **Real-time or streaming** operation with incremental diarization and partial hypotheses.
- **Richer enrollment**: explicit multi-sample enrollment, quality checks, and per-speaker score normalization.
- **Operator UI or dashboard** for reviewing segments, waveforms, and registry management.

---

## Project Structure

```text
Speaker-Diarization/
├── main.py                 # CLI entry and orchestration
├── requirements.txt
├── diarization/
│   └── diarize.py          # Pyannote diarization
├── asr/
│   └── transcribe.py       # WhisperX transcription and alignment
├── alignment/
│   └── align.py            # Overlap-based fusion and segment merging
├── embeddings/
│   └── speaker_id.py       # ECAPA-TDNN, registry I/O, open-set resolution
├── spoof_detection/
│   └── infer.py            # Wav2Vec2 spoof classifier
├── scripts/
│   └── enroll_speaker.py   # Enroll / list speakers in the registry
├── utils/
│   └── audio_utils.py      # Audio loading and resampling
├── data/
│   └── speaker_registry.json
├── pretrained_models/      # Created for SpeechBrain cache (e.g. ECAPA)
└── results/                # Default output JSON directory
```

---

## Citation and Licensing

If you use third-party models (pyannote, Whisper, SpeechBrain, Hugging Face weights), cite their respective licenses and terms of use. Add a project-level license file if you distribute this codebase publicly.
