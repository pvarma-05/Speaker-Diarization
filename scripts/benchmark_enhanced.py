"""
Enhanced speaker authorization benchmark.

Tests with:
- More speakers (15 total: 5 authorized, 10 unauthorized)
- Longer enrollment (4 clips averaged)
- More test clips per speaker (8)
- Noise augmentation (clean + noisy variants)

Usage: python scripts/benchmark_enhanced.py
"""

import os
import sys
import io
import tarfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TAR_PATH = os.path.join(DATA_DIR, "dev-clean.tar.gz")
BENCH_DIR = os.path.join(DATA_DIR, "benchmark_enhanced")

# 15 speakers from LibriSpeech dev-clean
ALL_SPEAKERS = {
    "84":   "Alice",    "174":  "Bob",      "1272": "Charlie",
    "1462": "Diana",    "1673": "Eve",
    "1919": "Frank",    "1988": "Grace",    "1993": "Hank",
    "2035": "Ivy",      "2078": "Jack",     "2086": "Karen",
    "2277": "Leo",      "2412": "Mia",      "2428": "Nick",
    "2803": "Olivia",
}

AUTHORIZED_IDS = ["84", "174", "1272", "1462", "1673"]  # 5 authorized
UNAUTHORIZED_IDS = [k for k in ALL_SPEAKERS if k not in AUTHORIZED_IDS]  # 10 unauthorized

ENROLLMENT_CLIPS = 4   # more clips for better voiceprint
TEST_CLIPS = 8         # more test clips
NEEDED = ENROLLMENT_CLIPS + TEST_CLIPS

# Noise levels (SNR in dB) — higher = less noise
NOISE_LEVELS = {
    "clean": None,
    "light_noise_20dB": 20,
    "moderate_noise_10dB": 10,
    "heavy_noise_5dB": 5,
}


def add_noise(audio, snr_db):
    """Add white noise at a given SNR level."""
    import numpy as np
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(audio.dtype)


def extract_speakers():
    """Extract clips for all speakers."""
    import soundfile as sf

    print("Extracting speaker clips from LibriSpeech...")
    os.makedirs(BENCH_DIR, exist_ok=True)

    clip_counts = {sid: 0 for sid in ALL_SPEAKERS}

    with tarfile.open(TAR_PATH, "r:gz") as tar:
        for member in tar.getmembers():
            parts = member.name.replace("\\", "/").split("/")
            if len(parts) < 4 or not member.name.endswith(".flac"):
                continue

            sid = parts[2]
            if sid not in ALL_SPEAKERS or clip_counts[sid] >= NEEDED:
                continue

            name = ALL_SPEAKERS[sid]
            spk_dir = os.path.join(BENCH_DIR, name)
            os.makedirs(spk_dir, exist_ok=True)

            existing = len([f for f in os.listdir(spk_dir) if f.endswith(".wav")])
            if existing >= NEEDED:
                clip_counts[sid] = existing
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            try:
                audio, sr = sf.read(io.BytesIO(f.read()))
                basename = os.path.splitext(os.path.basename(member.name))[0] + ".wav"
                sf.write(os.path.join(spk_dir, basename), audio, sr)
                clip_counts[sid] += 1
            except:
                continue

    for sid, name in ALL_SPEAKERS.items():
        auth = "AUTH" if sid in AUTHORIZED_IDS else "unauth"
        print(f"  {name:8s} (ID {sid:>4s}): {clip_counts[sid]} clips [{auth}]")

    return clip_counts


def run_benchmark():
    """Run the enhanced benchmark."""
    import numpy as np
    import soundfile as sf
    from embeddings.speaker_id import (
        load_speaker_encoder, extract_speaker_embedding,
        save_speaker_to_registry, load_speaker_registry,
        identify_speaker
    )

    encoder = load_speaker_encoder()
    registry_path = os.path.join(DATA_DIR, "benchmark_enhanced_registry.json")

    # --- Phase 1: Enroll authorized speakers ---
    print("\n--- Phase 1: Enrolling authorized speakers (4 clips each) ---")
    if os.path.exists(registry_path):
        os.remove(registry_path)

    for sid in AUTHORIZED_IDS:
        name = ALL_SPEAKERS[sid]
        spk_dir = os.path.join(BENCH_DIR, name)
        if not os.path.exists(spk_dir):
            print(f"  Warning: No data for {name}")
            continue
        clips = sorted([f for f in os.listdir(spk_dir) if f.endswith(".wav")])[:ENROLLMENT_CLIPS]

        embeddings = []
        for clip in clips:
            emb = extract_speaker_embedding(os.path.join(spk_dir, clip), encoder)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            save_speaker_to_registry(name, avg_emb, registry_path, {"role": "authorized"})
            print(f"  Enrolled {name} ({len(embeddings)} clips)")

    registry = load_speaker_registry(registry_path)

    # --- Phase 2: Test across noise conditions ---
    print("\n--- Phase 2: Testing identification ---")

    all_results = {}  # condition -> results list

    for noise_name, snr in NOISE_LEVELS.items():
        results = []
        all_speaker_ids = AUTHORIZED_IDS + UNAUTHORIZED_IDS

        for sid in all_speaker_ids:
            name = ALL_SPEAKERS[sid]
            is_auth = sid in AUTHORIZED_IDS
            spk_dir = os.path.join(BENCH_DIR, name)
            if not os.path.exists(spk_dir):
                continue
            clips = sorted([f for f in os.listdir(spk_dir) if f.endswith(".wav")])

            test_clips = clips[ENROLLMENT_CLIPS:ENROLLMENT_CLIPS + TEST_CLIPS]
            if not test_clips:
                test_clips = clips[-TEST_CLIPS:]

            for clip in test_clips:
                clip_path = os.path.join(spk_dir, clip)

                if snr is not None:
                    # Load, add noise, save to temp, extract embedding
                    audio, sr_rate = sf.read(clip_path)
                    noisy = add_noise(audio, snr)
                    tmp_path = os.path.join(BENCH_DIR, "_tmp_noisy.wav")
                    sf.write(tmp_path, noisy, sr_rate)
                    emb = extract_speaker_embedding(tmp_path, encoder)
                else:
                    emb = extract_speaker_embedding(clip_path, encoder)

                if emb is None:
                    continue

                pred_name, sim = identify_speaker(emb, registry, threshold=0.75, registry_path=registry_path)
                results.append({
                    "true_speaker": name,
                    "predicted": pred_name,
                    "similarity": sim,
                    "is_authorized": is_auth,
                })

        all_results[noise_name] = results

    # --- Phase 3: Compute and print metrics ---
    print("\n" + "=" * 70)
    print("ENHANCED BENCHMARK RESULTS")
    print(f"Speakers: {len(AUTHORIZED_IDS)} authorized, {len(UNAUTHORIZED_IDS)} unauthorized")
    print(f"Enrollment: {ENROLLMENT_CLIPS} clips per speaker | Test: {TEST_CLIPS} clips per speaker")
    print(f"Threshold: 0.75")
    print("=" * 70)

    summary_rows = []

    for noise_name, results in all_results.items():
        total = len(results)
        auth_clips = sum(1 for r in results if r["is_authorized"])
        unauth_clips = total - auth_clips

        tp = sum(1 for r in results if r["is_authorized"] and r["predicted"] == r["true_speaker"])
        fn = sum(1 for r in results if r["is_authorized"] and r["predicted"] != r["true_speaker"])
        tn = sum(1 for r in results if not r["is_authorized"] and r["predicted"] == "UNKNOWN")
        fp = sum(1 for r in results if not r["is_authorized"] and r["predicted"] != "UNKNOWN")

        acc = (tp + tn) / total * 100 if total > 0 else 0
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        far = fp / unauth_clips * 100 if unauth_clips > 0 else 0
        frr = fn / auth_clips * 100 if auth_clips > 0 else 0

        summary_rows.append((noise_name, total, acc, prec, rec, f1, far, frr, tp, tn, fp, fn))

    # Print summary table
    print(f"\n{'Condition':<22} {'Clips':<7} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'FAR':<8} {'FRR':<8}")
    print("-" * 77)
    for row in summary_rows:
        name, total, acc, prec, rec, f1, far, frr, tp, tn, fp, fn = row
        print(f"  {name:<20} {total:<7} {acc:<8.1f} {prec:<8.1f} {rec:<8.1f} {f1:<8.1f} {far:<8.1f} {frr:<8.1f}")

    # Print detailed confusion for each condition
    for noise_name, results in all_results.items():
        import numpy as np
        auth_sims = [r["similarity"] for r in results if r["is_authorized"]]
        unauth_sims = [r["similarity"] for r in results if not r["is_authorized"]]

        print(f"\n--- {noise_name} ---")
        if auth_sims:
            print(f"  Auth similarity:   avg={np.mean(auth_sims):.3f}  min={np.min(auth_sims):.3f}  max={np.max(auth_sims):.3f}")
        if unauth_sims:
            print(f"  Unauth similarity: avg={np.mean(unauth_sims):.3f}  min={np.min(unauth_sims):.3f}  max={np.max(unauth_sims):.3f}")

        # Show false accepts
        false_accepts = [r for r in results if not r["is_authorized"] and r["predicted"] != "UNKNOWN"]
        if false_accepts:
            print(f"  False Accepts ({len(false_accepts)}):")
            for r in false_accepts:
                print(f"    {r['true_speaker']} -> {r['predicted']} (sim={r['similarity']:.3f})")

        # Show false rejects
        false_rejects = [r for r in results if r["is_authorized"] and r["predicted"] != r["true_speaker"]]
        if false_rejects:
            print(f"  False Rejects ({len(false_rejects)}):")
            for r in false_rejects:
                print(f"    {r['true_speaker']} -> {r['predicted']} (sim={r['similarity']:.3f})")

    # Cleanup temp file
    tmp = os.path.join(BENCH_DIR, "_tmp_noisy.wav")
    if os.path.exists(tmp):
        os.remove(tmp)


if __name__ == "__main__":
    t0 = time.time()
    counts = extract_speakers()
    # Check if we have enough speakers
    valid = sum(1 for sid in ALL_SPEAKERS if counts.get(sid, 0) >= ENROLLMENT_CLIPS + 1)
    print(f"\nValid speakers with enough clips: {valid}/{len(ALL_SPEAKERS)}")
    run_benchmark()
    print(f"\nBenchmark completed in {time.time()-t0:.0f}s")
