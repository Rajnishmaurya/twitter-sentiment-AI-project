import os
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ================================
# Load and Prepare Data
# ================================

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_path = os.path.join(project_root, "data", "processed", "processed_data.csv")

df = pd.read_csv(data_path)
#df = df.dropna(subset=["clean_text", "sentiment"])
df = df.dropna(subset=["clean_text", "label"])


X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# ================================
# MLflow Tracking
# ================================

from pathlib import Path

mlruns_path = Path(project_root, "mlruns").as_uri()  # Converts to proper file URI
mlflow.set_tracking_uri(mlruns_path)


# mlflow.set_tracking_uri("file://" + os.path.join(project_root, "mlruns"))
# mlflow.set_experiment("TwitterSentimentAnalysis")

with mlflow.start_run(run_name="SVM_TFIDF"):
    mlflow.sklearn.autolog()

    # ================================
    # Pipeline: TF-IDF + SVM
    # ================================
    # pipeline = Pipeline([
    #     ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
    #     ("svm", SVC(kernel="linear", probability=True))
    # ])
    


    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, stop_words="english")),
        ("svm", LinearSVC())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log manually as well
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_param("model", "SVM")
    mlflow.log_param("vectorizer", "TF-IDF")

    # Save model + vectorizer
    processed_dir = os.path.join(project_root, "data", "processed")
    model_path = os.path.join(processed_dir, "svm_model.pkl")
    vectorizer_path = os.path.join(processed_dir, "tfidf_vectorizer.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(pipeline.named_steps["svm"], f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(pipeline.named_steps["tfidf"], f)

    mlflow.log_artifact(model_path, artifact_path="models")
    mlflow.log_artifact(vectorizer_path, artifact_path="vectorizer")

    print("✅ Model trained and logged to MLflow.")


# import os
# import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.metrics import classification_report, accuracy_score

# # === Paths ===
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
# MODEL_PATH = os.path.join(BASE_DIR, "data", "processed", "svm_model.pkl")
# VECTORIZER_PATH = os.path.join(BASE_DIR, "data", "processed", "tfidf_vectorizer.pkl")

# # === Load data ===
# df = pd.read_csv(DATA_PATH)

# # === Rename columns if needed ===
# if "sentiment" in df.columns and "label" not in df.columns:
#     df.rename(columns={"sentiment": "label"}, inplace=True)

# # === Drop missing rows ===
# df.dropna(subset=["clean_text", "label"], inplace=True)

# # === Split data ===
# X_train, X_test, y_train, y_test = train_test_split(
#     df["clean_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
# )

# # === TF-IDF + SVM Pipeline ===
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)

# model = LinearSVC()
# model.fit(X_train_vec, y_train)

# # === Evaluate ===
# y_pred = model.predict(X_test_vec)
# print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
# print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# # === Save model and vectorizer ===
# with open(MODEL_PATH, "wb") as f:
#     pickle.dump(model, f)

# with open(VECTORIZER_PATH, "wb") as f:
#     pickle.dump(vectorizer, f)

# print("✅ SVM model and TF-IDF vectorizer saved.")


