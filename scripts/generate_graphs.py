import pandas as pd
import matplotlib.pyplot as plt

csv_path = "static/outputs/crowd_counts.csv"

df = pd.read_csv(csv_path)

plt.figure(figsize=(12,5))

plt.plot(df["Frame"], df["Count"])

plt.xlabel("Frame")
plt.ylabel("Crowd Count")

plt.title("Crowd Density Over Time")

plt.grid(True)

plt.savefig("static/outputs/crowd_graph.png")

print("Graph Generated")