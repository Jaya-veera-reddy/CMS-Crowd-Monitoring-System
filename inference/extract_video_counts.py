import cv2
import torch
import csv
from models.csrnet import CSRNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
model = CSRNet().to(device)
model.load_state_dict(torch.load("weights/csrnet.pth", map_location=device))
model.eval()

video_path = "static/uploads/mall.mp4"

cap = cv2.VideoCapture(video_path)

frame_counts = []

frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img,(256,256))

    img = img / 255.0

    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        density = model(img)

    # scale factor due to resizing
    count = density.sum().item() * 64

    frame_counts.append([frame_id, count])

    print("Frame:", frame_id, "Count:", int(count))

    frame_id += 1


cap.release()

# Save results
with open("static/outputs/crowd_counts.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["frame","count"])

    writer.writerows(frame_counts)

print("Crowd counts saved to outputs/crowd_counts.csv")