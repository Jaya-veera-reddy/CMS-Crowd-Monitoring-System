from flask import Flask, render_template, request, redirect
import pandas as pd
import subprocess
import json
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =====================================================
# LOAD CSV DATA
# =====================================================

def load_csv_data():

    csv_path = "static/outputs/yolo_counts.csv"

    if os.path.exists(csv_path):

        df = pd.read_csv(csv_path)

        current_count = int(df["Current_Count"].iloc[-1])

        future_count = int(df["Predicted_5sec"].iloc[-1])
        crowd_history = df["Current_Count"].tolist()

        future_history = df["Predicted_5sec"].tolist()

        return (
            current_count,
            future_count,
            crowd_history,
            future_history
        )

    return 0, 0, [], []


# =====================================================
# LOAD RESULTS
# =====================================================

def load_results():

    results_path = "static/outputs/results.json"

    if os.path.exists(results_path):

        with open(results_path, "r") as f:

            results = json.load(f)

        return results

    return {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "mse": 0,
        "rmse": 0,
        "confusion_matrix": [[0,0],[0,0]]
    }


# =====================================================
# DASHBOARD
# =====================================================

@app.route('/')
def dashboard():

    current_count,
    future_count,
    crowd_history,
    future_history = load_csv_data()

    results = load_results()

    return render_template(
        'dashboard.html',

        current_count=current_count,
        future_count=future_count,

        crowd_history=crowd_history,
        future_history=future_history,

        accuracy=results['accuracy'],
        precision=results['precision'],
        recall=results['recall'],
        f1_score=results['f1_score'],
        mse=results['mse'],
        rmse=results['rmse'],
        confusion_matrix=results['confusion_matrix']
    )


# =====================================================
# VIDEO UPLOAD
# =====================================================

@app.route('/upload', methods=['POST'])
def upload_video():

    video = request.files['video']

    save_path = os.path.join(
        UPLOAD_FOLDER,
        'mall.mp4'
    )

    video.save(save_path)

    import sys

    subprocess.run([sys.executable, 'inference/yolo_count.py'])

    subprocess.run([sys.executable, 'training/evaluate_csrnet.py'])

    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)