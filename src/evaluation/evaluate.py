# import os
# from tensorflow.keras.models import load_model
# import numpy as np
# from sklearn.metrics import classification_report

# def evaluate_model():
#     #  This is the project root directory
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#     #  Corrected paths to match your actual structure
#     model_path = os.path.join(project_root, "data", "processed", "svm_model.pkl")
#     data_path = os.path.join(project_root, "data", "processed")

#     print(" Loading model and data...")
#     model = load_model(model_path)
#     X_test = np.load(os.path.join(data_path, "features.npy"))
#     y_test = np.load(os.path.join(data_path, "labels.npy"))

#     print(" Evaluating model...")
#     y_pred = model.predict(X_test)
#     y_pred_classes = np.argmax(y_pred, axis=1)
#     y_true_classes = y_test

#     print("\n Classification Report:")
#     print(classification_report(y_true_classes, y_pred_classes))

# if __name__ == "__main__":
#     evaluate_model()


import os
import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

def evaluate_model():
    # Set project root and paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(project_root, "data", "processed", "processed_data.csv")
    model_path = os.path.join(project_root, "data", "processed", "svm_model.pkl")
    vectorizer_path = os.path.join(project_root, "data", "processed", "tfidf_vectorizer.pkl")

    # Load processed data
    print(" Loading data...")
    df = pd.read_csv(data_path)
    df.dropna(subset=["clean_text", "label"], inplace=True)

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Load model and vectorizer
    print("🔍 Loading model and vectorizer...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Transform test data
    X_test_vec = vectorizer.transform(X_test)

    # Predict and evaluate
    print(" Evaluating model...")
    y_pred = model.predict(X_test_vec)

    print("\n Accuracy:", accuracy_score(y_test, y_pred))
    print("\n Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()
