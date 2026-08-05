import pandas as pd
import numpy as np

csv_path = "static/outputs/yolo_counts.csv"s

df = pd.read_csv(csv_path)

counts = df["Count"].values

current_count = int(counts[-1])

# Moving average prediction
future_predictions = []

window = counts[-10:]

avg_growth = np.mean(np.diff(window))

future = current_count

for i in range(5):

    future = future + avg_growth

    if future < 0:
        future = 0

    future_predictions.append(int(future))


print("\n==============================")
print("FUTURE CROWD PREDICTION")
print("==============================")

print("Current Crowd :", current_count)

print("\nFuture Forecast:")

for i, val in enumerate(future_predictions):

    print(f"Step {i+1}: {val}")


# SAVE RESULTS
with open("static/outputs/prediction.txt", "w") as f:

    f.write(f"Current Crowd: {current_count}\n\n")

    for i, val in enumerate(future_predictions):

        f.write(f"Future Step {i+1}: {val}\n")