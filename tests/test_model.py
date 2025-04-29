import os
import sys
import pickle
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "data", "processed", "svm_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "data", "processed", "tfidf_vectorizer.pkl")

def test_model_prediction():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    text = "I am so happy!"
    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)

    assert isinstance(prediction, np.ndarray) or isinstance(prediction, list)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]  # assuming binary classification

