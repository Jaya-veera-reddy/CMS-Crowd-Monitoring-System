import pandas as pd
import numpy as np

df = pd.read_csv("static/outputs/yolo_counts.csv")

counts = df["count"].values

window = 4

X = []
y = []

for i in range(len(counts) - window):

    X.append(counts[i:i+window])
    y.append(counts[i+window])

X = np.array(X)
y = np.array(y)

np.save("data/processed/X.npy", X)
np.save("data/processed/y.npy", y)

print("Timeseries dataset created")