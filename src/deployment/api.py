import os
import pickle
import sys
import logging
import asyncio
import subprocess
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator  # <-- add this

# Setup logging
logs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(logs_dir, "inference_logs.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Load model & vectorizer
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
processed_path = os.path.join(project_root, "data", "processed")
with open(os.path.join(processed_path, "svm_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(processed_path, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# Create FastAPI app
app = FastAPI()

# Mount Prometheus metrics at /metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, include_in_schema=False)

# Home page
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
        <title>Twitter Sentiment Prediction</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

            body {
                font-family: 'Poppins', sans-serif;
                background: linear-gradient(135deg, #74ebd5 0%, #acb6e5 100%);
                height: 100vh;
                margin: 0;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .container {
                background-color: white;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.2);
                text-align: center;
                width: 400px;
                animation: fadeIn 1s ease-in-out;
            }

            h2 {
                margin-bottom: 20px;
                color: #333;
            }

            textarea {
                width: 100%;
                height: 120px;
                padding: 10px;
                margin-top: 10px;
                margin-bottom: 20px;
                border-radius: 8px;
                border: 1px solid #ccc;
                font-size: 16px;
                resize: none;
                transition: 0.3s;
            }

            textarea:focus {
                border-color: #6c63ff;
                outline: none;
                box-shadow: 0px 0px 5px #6c63ff;
            }

            input[type=submit] {
                padding: 12px 24px;
                font-size: 16px;
                background-color: #6c63ff;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }

            input[type=submit]:hover {
                background-color: #574fd6;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-20px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🚀 Twitter Sentiment Prediction</h2>
            <form action="/predict" method="post">
                <textarea name="text" placeholder="Type your tweet here..."></textarea><br>
                <input type="submit" value="Predict Sentiment">
            </form>
        </div>
    </body>
    </html>
    """

# async def home():
#     return """
#     <html>
#       <body>
#         <h2>Twitter Sentiment Prediction</h2>
#         <form action="/predict" method="post">
#           <textarea name="text" rows="5" cols="40" placeholder="Type your tweet here..."></textarea><br><br>
#           <input type="submit" value="Predict Sentiment">
#         </form>
#       </body>
#     </html>
#     """

# Prediction endpoint
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    transformed = vectorizer.transform([text])
    
    # Get both prediction and probability
    #prob = model.predict_proba(transformed)[0]  # probability array
    prediction = model.predict(transformed)[0]

    #confidence = np.max(prob) * 100  # max probability %
    result = "Positive 😃" if prediction == 1 else "Negative 😞"

    logging.info(f"Input: {text} | Prediction: {result}")
    # Save text for drift detection
    recent_preds = os.path.join(logs_dir, "recent_predictions.txt")
    with open(recent_preds, "a", encoding="utf-8") as f:
        f.write(text + "\n")

    return f"""
    <html>
    <head>
        <title>Prediction Result</title>
        <style>
            body {{
                font-family: 'Poppins', sans-serif;
                background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                height: 100vh;
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .result-card {{
                background: white;
                padding: 40px 60px;
                border-radius: 12px;
                box-shadow: 0 8px 20px rgba(0,0,0,0.2);
                text-align: center;
                animation: popIn 0.6s ease-out;
            }}
            h2 {{
                color: #333;
                margin-bottom: 20px;
            }}
            h3 {{
                color: #6c63ff;
                margin: 10px 0;
            }}
            a {{
                text-decoration: none;
                color: white;
                background-color: #6c63ff;
                padding: 10px 20px;
                border-radius: 8px;
                margin-top: 20px;
                display: inline-block;
                transition: background-color 0.3s;
            }}
            a:hover {{
                background-color: #574fd6;
            }}
            @keyframes popIn {{
                from {{ transform: scale(0.5); opacity: 0; }}
                to {{ transform: scale(1); opacity: 1; }}
            }}
        </style>
    </head>
    <body>
        <div class="result-card">
            <h2>Prediction Result</h2>
            <h3>{result}</h3>
            <a href="/">🔙 Go Back</a>
        </div>
    </body>
    </html>
    """

# @app.post("/predict")
# async def predict(request: Request, text: str = Form(...)):
#     transformed = vectorizer.transform([text])
#     prediction = model.predict(transformed)[0]
#     result = "Positive" if prediction == 1 else "Negative"
#     logging.info(f"Input: {text} | Prediction: {result}")

#     # Save text for drift detection
#     recent_preds = os.path.join(logs_dir, "recent_predictions.txt")
#     with open(recent_preds, "a", encoding="utf-8") as f:
#         f.write(text + "\n")

#     return HTMLResponse(f"<h3>Sentiment: {result}</h3><br><a href='/'>Back</a>")

# Background drift monitor (runs every 20 seconds)
async def monitor_drift():
    print(" Drift monitor started", flush=True)
    drift_script = os.path.join(project_root, "src", "drift", "drift_detection.py")
    while True:
        print("🔍 Checking for data drift...", flush=True)
        res = subprocess.run([sys.executable, drift_script], capture_output=True, text=True)
        if res.stdout: print(res.stdout, flush=True)
        if res.stderr: print(" Drift error:", res.stderr, flush=True)
        await asyncio.sleep(20)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(monitor_drift())

# working code

# import os
# import pickle
# import sys
# import logging
# import asyncio
# import subprocess
# from fastapi import FastAPI, Form, Request
# from fastapi.responses import HTMLResponse

# # Setup logging with timestamp
# logs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
# os.makedirs(logs_dir, exist_ok=True)

# logging.basicConfig(
#     filename=os.path.join(logs_dir, "inference_logs.log"),
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# )

# # ─────────────────────────────────────────────────────────────────────────────
# # Load model and vectorizer
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# processed_path = os.path.join(project_root, "data", "processed")

# with open(os.path.join(processed_path, "svm_model.pkl"), "rb") as f:
#     model = pickle.load(f)

# with open(os.path.join(processed_path, "tfidf_vectorizer.pkl"), "rb") as f:
#     vectorizer = pickle.load(f)

# # ─────────────────────────────────────────────────────────────────────────────
# # FastAPI app
# app = FastAPI()

# # Root endpoint
# @app.get("/", response_class=HTMLResponse)
# async def home():
#     return """
#     <html>
#         <body>
#             <h2>Twitter Sentiment Prediction</h2>
#             <form action="/predict" method="post">
#                 <textarea name="text" rows="5" cols="40" placeholder="Type your tweet here..."></textarea><br><br>
#                 <input type="submit" value="Predict Sentiment">
#             </form>
#         </body>
#     </html>
#     """

# # Prediction endpoint
# @app.post("/predict")
# async def predict(request: Request, text: str = Form(...)):
#     transformed = vectorizer.transform([text])
#     prediction = model.predict(transformed)[0]
#     result = "Positive" if prediction == 1 else "Negative"

#     # Log the prediction
#     logging.info(f"Input: {text} | Prediction: {result}")

#     # Save the recent prediction to 'recent_predictions.txt'
#     recent_predictions_path = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "recent_predictions.txt")
    
#     # Check if the file exists and create it if it doesn't
#     if not os.path.exists(recent_predictions_path):
#         with open(recent_predictions_path, "w", encoding="utf-8") as f:
#             f.write(f"{text} | Sentiment: {result}\n")
#     else:
#         # Append the prediction to the file
#         with open(recent_predictions_path, "a", encoding="utf-8") as f:
#             f.write(f"{text} | Sentiment: {result}\n")

#     return HTMLResponse(f"<h3>Sentiment: {result}</h3><br><a href='/'>Back</a>")

# # ─────────────────────────────────────────────────────────────────────────────
# # Drift Detection Function
# async def monitor_drift():
#     print("✅ Drift monitor task started", flush=True)
#     while True:
#         print("🔍 Checking for data drift...", flush=True)
#         # Run the drift script
#         result = subprocess.run(
#             [sys.executable, os.path.join(project_root, "src", "drift", "drift_detection.py")],
#             capture_output=True,
#             text=True
#         )
#         # Print both stdout and stderr from the drift detection script
#         if result.stdout:
#             print(result.stdout, flush=True)
#         if result.stderr:
#             print("🛑 Drift script error:\n", result.stderr, flush=True)
#         await asyncio.sleep(20)  # Check every 20 seconds

# # ─────────────────────────────────────────────────────────────────────────────
# # FastAPI Event - Run drift detection in the background when the app starts
# @app.on_event("startup")
# async def startup_event():
#     asyncio.create_task(monitor_drift())
