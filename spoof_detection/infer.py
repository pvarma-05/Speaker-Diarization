"""Wav2vec2-based deepfake/spoof audio detector."""

import numpy as np
import torch
import librosa

WAV2VEC2_MODEL_ID = "mo-thecreator/Deepfake-audio-detection"


class SpoofDetector:
    """Wav2vec2-based deepfake/spoof audio detector.

    Uses a fine-tuned wav2vec2-base model from HuggingFace for
    classifying audio segments as real (human) or fake (AI/TTS).
    The model is downloaded automatically on first use (~378 MB).
    """

    def __init__(self, model_id=None):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        model_id = model_id or WAV2VEC2_MODEL_ID
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self._model = AutoModelForAudioClassification.from_pretrained(model_id).to(self._device)
        self._model.eval()

    def predict(self, audio_path_or_array, sr=16000):
        """Classify an audio segment as Real or Spoof.

        Args:
            audio_path_or_array: File path (str) or numpy array of audio.
            sr: Sample rate (default 16000).

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

        inputs = self._extractor(
            X.astype(np.float32), sampling_rate=sr, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        predicted_class = probs.argmax(-1).item()
        score = probs[0, predicted_class].item()
        label_raw = self._model.config.id2label[predicted_class].lower()

        if "fake" in label_raw or "spoof" in label_raw:
            return "Spoof", score
        else:
            return "Real", score


if __name__ == "__main__":
    detector = SpoofDetector()
    print("Model loaded successfully.")
