
import numpy as np
import torch
import librosa

# HuggingFace model ID for wav2vec2-based deepfake detection
WAV2VEC2_MODEL_ID = "mo-thecreator/Deepfake-audio-detection"


class SpoofDetector:
    """Wav2vec2-based deepfake/spoof audio detector.

    Uses a fine-tuned wav2vec2-base model from HuggingFace for
    classifying audio segments as real (human) or fake (AI/TTS).
    The model is downloaded automatically on first use (~378 MB).
    """

    def __init__(self, model_id=None):
        from transformers import pipeline as hf_pipeline

        model_id = model_id or WAV2VEC2_MODEL_ID
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = hf_pipeline(
            "audio-classification",
            model=model_id,
            device=device,
        )
        print(f"Loaded Anti-Spoofing Model: {model_id}")

    def predict(self, audio_path_or_array, sr=16000):
        """Classify an audio segment as Real or Spoof.

        Args:
            audio_path_or_array: File path (str) or numpy array of audio.
            sr: Sample rate (default 16000). Audio will be resampled if needed.

        Returns:
            Tuple of (label, score) where label is "Real" or "Spoof"
            and score is the confidence (0.0 to 1.0).
        """
        if isinstance(audio_path_or_array, str):
            X, sr = librosa.load(audio_path_or_array, sr=16000)
        else:
            X = audio_path_or_array
            if sr != 16000:
                X = librosa.resample(X, orig_sr=sr, target_sr=16000)
                sr = 16000

        result = self._pipe({"raw": X.astype(np.float32), "sampling_rate": sr})
        top = result[0]
        label_raw = top["label"].lower()
        score = top["score"]

        if "fake" in label_raw or "spoof" in label_raw:
            return "Spoof", score
        else:
            return "Real", score


if __name__ == "__main__":
    detector = SpoofDetector()
    print("Model loaded successfully.")
