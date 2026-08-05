import pandas as pd

df = pd.read_csv("static/outputs/crowd_counts.csv")

avg_count = df["Count"].mean()

max_count = df["Count"].max()

report = f"""
Crowd Analysis Report
=====================

Average Crowd Count : {avg_count:.2f}

Maximum Crowd Count : {max_count}

"""

if max_count > 50:
    report += "\nHIGH CROWD DENSITY"

elif max_count > 30:
    report += "\nMEDIUM CROWD DENSITY"

else:
    report += "\nLOW CROWD DENSITY"


with open("static/outputs/report.txt", "w") as f:
    f.write(report)

print("Report Generated")