"""
Comprehensive speaker authorization benchmark.

Extracts 10 speakers from LibriSpeech, enrolls 5 as authorized,
tests identification on separate clips, and computes detailed metrics.

Usage: python scripts/benchmark_auth.py
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
BENCH_DIR = os.path.join(DATA_DIR, "benchmark_speakers")

# 10 speakers from LibriSpeech dev-clean
ALL_SPEAKERS = {
    "84":   "Speaker_A",
    "174":  "Speaker_B",
    "1272": "Speaker_C",
    "1462": "Speaker_D",
    "1673": "Speaker_E",
    "1919": "Speaker_F",
    "1988": "Speaker_G",
    "1993": "Speaker_H",
    "2035": "Speaker_I",
    "2078": "Speaker_J",
}

# First 5 = authorized, last 5 = unauthorized
AUTHORIZED_IDS = ["84", "174", "1272", "1462", "1673"]
UNAUTHORIZED_IDS = ["1919", "1988", "1993", "2035", "2078"]

ENROLLMENT_CLIPS = 2   # clips used for enrollment
TEST_CLIPS = 5         # clips used for testing (separate from enrollment)


def extract_benchmark_speakers():
    """Extract clips for all 10 speakers."""
    import soundfile as sf

    print("Extracting speaker clips from LibriSpeech...")
    os.makedirs(BENCH_DIR, exist_ok=True)

    # Track how many clips per speaker we've extracted
    clip_counts = {sid: 0 for sid in ALL_SPEAKERS}
    needed = ENROLLMENT_CLIPS + TEST_CLIPS

    with tarfile.open(TAR_PATH, "r:gz") as tar:
        for member in tar.getmembers():
            parts = member.name.replace("\\", "/").split("/")
            if len(parts) < 4 or not member.name.endswith(".flac"):
                continue

            sid = parts[2]
            if sid not in ALL_SPEAKERS or clip_counts[sid] >= needed:
                continue

            name = ALL_SPEAKERS[sid]
            spk_dir = os.path.join(BENCH_DIR, name)
            os.makedirs(spk_dir, exist_ok=True)

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
        print(f"  {name} (ID {sid}): {clip_counts[sid]} clips [{auth}]")


def run_benchmark():
    """Run the full benchmark."""
    import numpy as np
    import soundfile as sf
    from embeddings.speaker_id import (
        load_speaker_encoder, extract_speaker_embedding,
        save_speaker_to_registry, load_speaker_registry,
        identify_speaker, cosine_similarity
    )

    encoder = load_speaker_encoder()
    registry_path = os.path.join(DATA_DIR, "benchmark_registry.json")

    # --- Phase 1: Enroll authorized speakers ---
    print("\n--- Phase 1: Enrolling authorized speakers ---")
    if os.path.exists(registry_path):
        os.remove(registry_path)

    for sid in AUTHORIZED_IDS:
        name = ALL_SPEAKERS[sid]
        spk_dir = os.path.join(BENCH_DIR, name)
        clips = sorted([f for f in os.listdir(spk_dir) if f.endswith(".wav")])[:ENROLLMENT_CLIPS]

        # Average embedding across enrollment clips
        embeddings = []
        for clip in clips:
            emb = extract_speaker_embedding(os.path.join(spk_dir, clip), encoder)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            save_speaker_to_registry(name, avg_emb, registry_path, {"role": "authorized"})
            print(f"  ✓ Enrolled {name} ({len(embeddings)} clips)")

    registry = load_speaker_registry(registry_path)

    # --- Phase 2: Test identification ---
    print("\n--- Phase 2: Testing identification ---")
    results = []  # (true_speaker, predicted_speaker, similarity, is_authorized_ground_truth)

    all_speaker_ids = AUTHORIZED_IDS + UNAUTHORIZED_IDS
    for sid in all_speaker_ids:
        name = ALL_SPEAKERS[sid]
        is_auth = sid in AUTHORIZED_IDS
        spk_dir = os.path.join(BENCH_DIR, name)
        clips = sorted([f for f in os.listdir(spk_dir) if f.endswith(".wav")])

        # Use clips AFTER enrollment clips for testing
        test_clips = clips[ENROLLMENT_CLIPS:ENROLLMENT_CLIPS + TEST_CLIPS]
        if not test_clips:
            test_clips = clips[-TEST_CLIPS:]  # fallback

        for clip in test_clips:
            emb = extract_speaker_embedding(os.path.join(spk_dir, clip), encoder)
            if emb is None:
                continue

            pred_name, sim = identify_speaker(emb, registry, threshold=0.75, registry_path=registry_path)
            results.append({
                "true_speaker": name,
                "predicted": pred_name,
                "similarity": sim,
                "is_authorized": is_auth,
                "would_include": pred_name != "UNKNOWN",
            })

    # --- Phase 3: Compute metrics ---
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    total = len(results)
    print(f"\nTotal test clips: {total}")
    print(f"Authorized speakers: {len(AUTHORIZED_IDS)} ({ENROLLMENT_CLIPS} enrollment + {TEST_CLIPS} test clips each)")
    print(f"Unauthorized speakers: {len(UNAUTHORIZED_IDS)} ({TEST_CLIPS} test clips each)")

    # --- Per-speaker results ---
    print(f"\n{'Speaker':<12} {'Auth?':<8} {'Predicted':<12} {'Sim':<6} {'Result'}")
    print("-" * 55)

    for r in results:
        auth_str = "AUTH" if r["is_authorized"] else "unauth"

        if r["is_authorized"]:
            if r["predicted"] == r["true_speaker"]:
                result = "✓ Correct"
            elif r["predicted"] == "UNKNOWN":
                result = "✗ False Reject"
            else:
                result = f"✗ Misidentified as {r['predicted']}"
        else:
            if r["predicted"] == "UNKNOWN":
                result = "✓ Blocked"
            else:
                result = f"✗ False Accept (as {r['predicted']})"

        print(f"  {r['true_speaker']:<12} {auth_str:<8} {r['predicted']:<12} {r['similarity']:.2f}  {result}")

    # --- Aggregate metrics ---
    # True Positive: authorized speaker correctly identified
    tp = sum(1 for r in results if r["is_authorized"] and r["predicted"] == r["true_speaker"])
    # False Negative: authorized speaker rejected (predicted UNKNOWN or wrong name)
    fn = sum(1 for r in results if r["is_authorized"] and r["predicted"] != r["true_speaker"])
    # True Negative: unauthorized speaker correctly blocked
    tn = sum(1 for r in results if not r["is_authorized"] and r["predicted"] == "UNKNOWN")
    # False Positive: unauthorized speaker incorrectly accepted
    fp = sum(1 for r in results if not r["is_authorized"] and r["predicted"] != "UNKNOWN")

    auth_clips = sum(1 for r in results if r["is_authorized"])
    unauth_clips = sum(1 for r in results if not r["is_authorized"])

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    far = fp / unauth_clips * 100 if unauth_clips > 0 else 0  # False Acceptance Rate
    frr = fn / auth_clips * 100 if auth_clips > 0 else 0      # False Rejection Rate

    print(f"\n{'='*60}")
    print(f"AGGREGATE METRICS")
    print(f"{'='*60}")
    print(f"  Overall Accuracy:         {accuracy:.1f}%  ({tp+tn}/{total})")
    print(f"  Precision:                {precision:.1f}%  (of accepted, how many are authorized)")
    print(f"  Recall:                   {recall:.1f}%  (of authorized, how many were accepted)")
    print(f"  F1 Score:                 {f1:.1f}%")
    print(f"  False Acceptance Rate:    {far:.1f}%  ({fp}/{unauth_clips} unauthorized got through)")
    print(f"  False Rejection Rate:     {frr:.1f}%  ({fn}/{auth_clips} authorized were blocked)")
    print()
    print(f"  True Positives:  {tp}   (authorized correctly identified)")
    print(f"  True Negatives:  {tn}   (unauthorized correctly blocked)")
    print(f"  False Positives: {fp}   (unauthorized incorrectly accepted)")
    print(f"  False Negatives: {fn}   (authorized incorrectly rejected)")

    # --- Similarity score analysis ---
    auth_sims = [r["similarity"] for r in results if r["is_authorized"]]
    unauth_sims = [r["similarity"] for r in results if not r["is_authorized"]]

    print(f"\n{'='*60}")
    print(f"SIMILARITY SCORE ANALYSIS (threshold = 0.75)")
    print(f"{'='*60}")
    if auth_sims:
        print(f"  Authorized speakers:   avg={np.mean(auth_sims):.3f}  min={np.min(auth_sims):.3f}  max={np.max(auth_sims):.3f}")
    if unauth_sims:
        print(f"  Unauthorized speakers: avg={np.mean(unauth_sims):.3f}  min={np.min(unauth_sims):.3f}  max={np.max(unauth_sims):.3f}")

    # --- Threshold sensitivity ---
    print(f"\n{'='*60}")
    print(f"THRESHOLD SENSITIVITY")
    print(f"{'='*60}")
    print(f"  {'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'FAR':<10} {'FRR':<10}")

    for thresh in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        t_tp = sum(1 for r in results if r["is_authorized"] and r["similarity"] >= thresh
                   and r["predicted"] != "UNKNOWN")
        t_fn = auth_clips - t_tp
        t_fp = sum(1 for r in results if not r["is_authorized"] and r["similarity"] >= thresh
                   and r["predicted"] != "UNKNOWN")
        t_tn = unauth_clips - t_fp
        t_acc = (t_tp + t_tn) / total * 100
        t_prec = t_tp / (t_tp + t_fp) * 100 if (t_tp + t_fp) > 0 else 0
        t_rec = t_tp / (t_tp + t_fn) * 100 if (t_tp + t_fn) > 0 else 0
        t_far = t_fp / unauth_clips * 100
        t_frr = t_fn / auth_clips * 100
        print(f"  {thresh:<12.2f} {t_acc:<10.1f} {t_prec:<10.1f} {t_rec:<10.1f} {t_far:<10.1f} {t_frr:<10.1f}")


if __name__ == "__main__":
    t0 = time.time()
    extract_benchmark_speakers()
    run_benchmark()
    print(f"\nBenchmark completed in {time.time()-t0:.0f}s")
