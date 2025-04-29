import os
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from prometheus_client import Gauge, CollectorRegistry, push_to_gateway

# ─── Setup Prometheus Metric ───────────────────────────
registry = CollectorRegistry()
drift_metric = Gauge('data_drift_detected', 'Data Drift Detection Status', registry=registry)

# ─── Paths ─────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
processed_dir = os.path.join(project_root, "data", "processed")
logs_dir      = os.path.join(project_root, "logs")

vectorizer_path          = os.path.join(processed_dir, "tfidf_vectorizer.pkl")
training_csv_path        = os.path.join(processed_dir, "processed_data.csv")
recent_predictions_path  = os.path.join(logs_dir, "recent_predictions.txt")

# ─── Load Texts ─────────────────────────────────────────
def load_texts():
    # Load vectorizer
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Load training texts
    df = pd.read_csv(training_csv_path)
    train_texts = df["clean_text"].dropna().tolist()

    # Load recent prediction texts
    if os.path.exists(recent_predictions_path):
        with open(recent_predictions_path, "r", encoding="utf-8") as f:
            recent_texts = [line.strip() for line in f if line.strip()]
    else:
        recent_texts = []

    return vectorizer, train_texts, recent_texts

# ─── Drift Detection ────────────────────────────────────
def detect_drift():
    vectorizer, train_texts, recent_texts = load_texts()

    # Check counts
    if len(train_texts) < 10:
        print(" Drift Check Aborted: Need ≥10 training samples.", flush=True)
        return
    if len(recent_texts) < 10:
        print(f" Drift Check Skipped: Need ≥10 recent samples. Got {len(recent_texts)}.", flush=True)
        return

    # Vectorize
    X_train  = vectorizer.transform(train_texts)
    X_recent = vectorizer.transform(recent_texts)

    # Compute centroids
    c_train  = np.asarray(X_train.mean(axis=0)).flatten()
    c_recent = np.asarray(X_recent.mean(axis=0)).flatten()

    # Drift score
    score = cosine(c_train, c_recent)
    print(f" Drift score (cosine distance): {score:.4f}", flush=True)

    # Drift decision
    THRESHOLD = 0.3
    if score > THRESHOLD:
        print(" Drift detected! (score > threshold)", flush=True)
        drift_metric.set(1)
    else:
        print(" No significant drift detected.", flush=True)
        drift_metric.set(0)

    # Push metric to Pushgateway
    try:
        push_to_gateway('pushgateway:9091', job='data_drift_job', registry=registry)
        print(" Drift metric successfully pushed to Pushgateway!", flush=True)
    except Exception as e:
        print(f" Failed to push drift metric: {e}", flush=True)

# ─── Main Execution ─────────────────────────────────────
if __name__ == "__main__":
    detect_drift()


# # src/drift/drift_detection.py

# import os
# import pickle
# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import cosine

# from prometheus_client import Gauge, CollectorRegistry, push_to_gateway

# # Setup Prometheus registry
# registry = CollectorRegistry()
# drift_metric = Gauge('data_drift_detected', 'Data Drift Detection Status', registry=registry)

# # Pushgateway address
# PUSHGATEWAY_ADDRESS = 'http://pushgateway:9091'

# def load_texts():
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#     processed_dir = os.path.join(project_root, "data", "processed")
#     logs_dir = os.path.join(project_root, "logs")

#     vec_path    = os.path.join(processed_dir, "tfidf_vectorizer.pkl")
#     train_path  = os.path.join(processed_dir, "processed_data.csv")
#     recent_path = os.path.join(logs_dir, "recent_predictions.txt")

#     # Load vectorizer
#     with open(vec_path, "rb") as f:
#         vectorizer = pickle.load(f)

#     # Load training texts
#     df = pd.read_csv(train_path)
#     train_texts = df["clean_text"].dropna().tolist()

#     # Load recent prediction texts
#     if os.path.exists(recent_path):
#         with open(recent_path, "r", encoding="utf-8") as f:
#             recent = [l.strip() for l in f if l.strip()]
#     else:
#         recent = []

#     return vectorizer, train_texts, recent

# def detect_drift():
#     vectorizer, train_texts, recent_texts = load_texts()
    
#     # Check counts
#     if len(train_texts) < 10:
#         print(" Drift Check Aborted: need ≥10 training samples", flush=True)
#         return
#     if len(recent_texts) < 10:
#         print(f" Drift Check Skipped: need ≥10 recent samples, got {len(recent_texts)}", flush=True)
#         return

#     # Vectorize
#     X_train  = vectorizer.transform(train_texts)
#     X_recent = vectorizer.transform(recent_texts)

#     # Compute centroids
#     c_train  = np.asarray(X_train.mean(axis=0)).flatten()
#     c_recent = np.asarray(X_recent.mean(axis=0)).flatten()

#     # Drift score
#     score = cosine(c_train, c_recent)
#     print(f" Drift score: {score:.4f}", flush=True)

#     # Decision
#     THRESHOLD = 0.3
#     if score > THRESHOLD:
#         print("⚠️ Drift detected! (score > threshold)", flush=True)
#         drift_metric.set(1)  # Drift occurred
#     else:
#         print("✅ No significant drift detected.", flush=True)
#         drift_metric.set(0)  # No drift

#     # 👇 Push metric to Pushgateway (after setting value)
#     push_to_gateway(PUSHGATEWAY_ADDRESS, job='data_drift_job', registry=registry)

# if __name__ == "__main__":
#     detect_drift()


# import os
# import sys
# import pickle
# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import cosine

# # ─── Paths ────────────────────────────────────────────────────────────────────
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# processed_dir = os.path.join(project_root, "data", "processed")
# logs_dir      = os.path.join(project_root, "logs")

# vectorizer_path          = os.path.join(processed_dir, "tfidf_vectorizer.pkl")
# training_csv_path        = os.path.join(processed_dir, "processed_data.csv")
# recent_predictions_path  = os.path.join(logs_dir, "recent_predictions.txt")

# # ─── Load TF-IDF Vectorizer ───────────────────────────────────────────────────
# with open(vectorizer_path, "rb") as f:
#     vectorizer = pickle.load(f)

# # ─── Load Training Texts ──────────────────────────────────────────────────────
# df_train = pd.read_csv(training_csv_path)
# train_texts = df_train["clean_text"].dropna().tolist()
# if len(train_texts) < 10:
#     print("❌ Not enough training texts for vectorizer.")
#     sys.exit(1)

# # ─── Load Recent Prediction Texts ────────────────────────────────────────────
# if not os.path.exists(recent_predictions_path):
#     print("❌ No recent_predictions.txt found. Run some predictions first.")
#     sys.exit(1)

# with open(recent_predictions_path, "r", encoding="utf-8") as f:
#     recent_texts = [line.strip() for line in f if line.strip()]

# if len(recent_texts) < 10:
#     print(f"⏳ Need at least 10 recent samples; got {len(recent_texts)}. Skipping drift check.")
#     sys.exit(0)

# # ─── Vectorize ────────────────────────────────────────────────────────────────
# X_train  = vectorizer.transform(train_texts)
# X_recent = vectorizer.transform(recent_texts)

# # ─── Compute Centroids ────────────────────────────────────────────────────────
# train_centroid  = np.asarray(X_train.mean(axis=0)).flatten()
# recent_centroid = np.asarray(X_recent.mean(axis=0)).flatten()

# # ─── Compute Drift Score ──────────────────────────────────────────────────────
# drift_score = cosine(train_centroid, recent_centroid)
# print(f"🔍 Drift score (cosine distance): {drift_score:.4f}")

# # ─── Threshold & Action ───────────────────────────────────────────────────────
# THRESHOLD = 0.3
# if drift_score > THRESHOLD:
#     print("⚠️ Drift detected! (score > threshold)")
#     # Trigger your DVC pipeline retrain
#     #os.system("dvc repro")
# else:
#     print("✅ No significant drift detected.")
