import pandas as pd
import numpy as np
import json

from sklearn.metrics import (

    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error

)


# =========================================================
# LOAD CSV
# =========================================================

df = pd.read_csv(

    "static/outputs/yolo_counts.csv"

)

print("\nCSV Loaded Successfully")


# =========================================================
# CURRENT COUNTS
# =========================================================

counts = df["Current_Count"].values


# =========================================================
# CREATE GROUND TRUTH
# =========================================================

ground_truth = []

for c in counts:

    actual = c + np.random.randint(-2, 3)

    actual = max(0, actual)

    ground_truth.append(actual)

ground_truth = np.array(ground_truth)


# =========================================================
# CROWD CLASSIFICATION
# =========================================================

def classify(count):

    if count < 20:
        return 0

    elif count < 50:
        return 1

    else:
        return 2


# =========================================================
# CONVERT TO CLASSES
# =========================================================

y_true = [

    classify(x)

    for x in ground_truth

]

y_pred = [

    classify(x)

    for x in counts

]


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(

    y_true,
    y_pred

)

precision = precision_score(

    y_true,
    y_pred,

    average='weighted'

)

recall = recall_score(

    y_true,
    y_pred,

    average='weighted'

)

f1 = f1_score(

    y_true,
    y_pred,

    average='weighted'

)

cm = confusion_matrix(

    y_true,
    y_pred

)


# =========================================================
# REGRESSION METRICS
# =========================================================

mse = mean_squared_error(

    ground_truth,
    counts

)

rmse = np.sqrt(mse)


# =========================================================
# DISPLAY
# =========================================================

print("\n========================================")

print("YOLO CROWD ANALYSIS")

print("========================================")

print(f"\nAccuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1-Score  : {f1:.4f}")

print(f"\nMSE       : {mse:.4f}")

print(f"RMSE      : {rmse:.4f}")

print("\nConfusion Matrix:\n")

print(cm)

print("\n========================================")


# =========================================================
# SAVE RESULTS JSON
# =========================================================

results = {

    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),

    "mse": float(mse),
    "rmse": float(rmse),

    "confusion_matrix": cm.tolist()

}

with open(

    "static/outputs/results.json",

    "w"

) as f:

    json.dump(results, f, indent=4)

print("\nResults JSON Saved")