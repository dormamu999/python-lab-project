import os
import pickle
import numpy as np
import soundfile as sf
from scipy.spatial.distance import cosine
import time

# Fast fingerprint using downsampled raw audio
def load_and_fingerprint(path, duration=1.0):
    try:
        data, sr = sf.read(path, dtype='float32')
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)  # Convert stereo to mono
        data = data[:int(sr * duration)]
        if len(data) < 100:
            return None
        fingerprint = np.mean(data.reshape(-1, 100), axis=1)
        fingerprint = fingerprint / np.linalg.norm(fingerprint)  # Normalize
        return fingerprint
    except Exception as e:
        print(f"❌ Failed: {path} - {e}")
        return None

# Build or load database
def build_db(audio_dir, db_path):
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            return pickle.load(f)

    db = {}
    for file in os.listdir(audio_dir):
        if file.lower().endswith(".wav"):
            path = os.path.join(audio_dir, file)
            print(f"📦 Fingerprinting: {file}")
            fp = load_and_fingerprint(path)
            if fp is not None:
                db[file] = fp

    with open(db_path, 'wb') as f:
        pickle.dump(db, f)
    return db

# Match test fingerprint
def match(test_fp, db):
    best_match, best_score = None, float("inf")
    for name, fp in db.items():
        score = cosine(test_fp, fp)
        print(f"🔍 {name}: similarity = {1 - score:.3f}")
        if score < best_score:
            best_score = score
            best_match = name
    return best_match, best_score

# Main recognition
def recognize(test_file):
    audio_dir = r"C:\Users\vinit\OneDrive\Desktop\python lab project\hope\songs"
    db_path = os.path.join(audio_dir, "fast_db.pkl")

    if not os.path.exists(test_file):
        print("❌ Test file not found.")
        return

    start = time.time()
    db = build_db(audio_dir, db_path)
    test_fp = load_and_fingerprint(test_file)
    if test_fp is None:
        print("❌ Could not read test file properly.")
        return

    match_file, score = match(test_fp, db)

    print("\n🎤 RESULT:")
    if match_file:
        print(f"✅ BEST MATCH: {match_file} (Similarity Score: {1 - score:.3f})")
    else:
        print("❌ No match found.")
    print(f"\n⏱ Completed in {time.time() - start:.2f}s")

# Run
if __name__ == "__main__":
    test_file = r"C:\Users\vinit\OneDrive\Desktop\zest.wav"
    recognize(test_file)
