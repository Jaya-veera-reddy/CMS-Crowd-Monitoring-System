import cv2
import csv
import os
import numpy as np

from ultralytics import YOLO


# =========================================================
# LOAD YOLO MODEL
# =========================================================

model = YOLO("yolov8n.pt")

print("\nYOLOv8 Model Loaded Successfully")


# =========================================================
# VIDEO PATH
# =========================================================

video_path = "static/uploads/mall.mp4"

print("\nProcessing Video:", video_path)


# =========================================================
# CHECK VIDEO
# =========================================================

if not os.path.exists(video_path):

    print("\nVideo not found!")
    exit()


# =========================================================
# OUTPUTS
# =========================================================

os.makedirs("static/outputs", exist_ok=True)

output_video = "static/outputs/yolo_output.mp4"

csv_path = "static/outputs/yolo_counts.csv"


# =========================================================
# VIDEO CAPTURE
# =========================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    print("\nError opening video!")
    exit()


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(cap.get(cv2.CAP_PROP_FPS))


# =========================================================
# VIDEO WRITER
# =========================================================

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(

    output_video,

    fourcc,

    fps,

    (width, height)

)


# =========================================================
# CSV
# =========================================================

csv_file = open(csv_path, mode='w', newline='')

writer = csv.writer(csv_file)

writer.writerow([

    "Frame",
    "Current_Count",
    "Predicted_5sec",
    "Alert"

])


# =========================================================
# VARIABLES
# =========================================================

frame_num = 0

count_history = []


# =========================================================
# PROCESS VIDEO
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # =====================================================
    # YOLO DETECTION
    # =====================================================

    results = model(frame)


    # =====================================================
    # PERSON COUNT
    # =====================================================

    person_count = 0


    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            confidence = float(box.conf[0])

            # PERSON CLASS = 0
            if cls == 0 and confidence > 0.4:

                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2

                )


    # =====================================================
    # STORE HISTORY
    # =====================================================

    count_history.append(person_count)


    # =====================================================
    # KEEP LAST 30 COUNTS
    # =====================================================

    if len(count_history) > 30:

        count_history.pop(0)


    # =====================================================
    # SIMPLE FUTURE PREDICTION
    # =====================================================

    if len(count_history) >= 5:

        trend = np.mean(np.diff(count_history[-5:]))

        predicted_5sec = int(

            person_count + (trend * 5)

        )

    else:

        predicted_5sec = person_count


    predicted_5sec = max(0, predicted_5sec)


    # =====================================================
    # ALERTS
    # =====================================================

    if predicted_5sec > 80:

        alert = "HIGH CROWD ALERT"

        color = (0, 0, 255)

    elif predicted_5sec > 40:

        alert = "MEDIUM CROWD"

        color = (0, 255, 255)

    else:

        alert = "NORMAL"

        color = (0, 255, 0)


    # =====================================================
    # WRITE CSV
    # =====================================================

    writer.writerow([

        frame_num,
        person_count,
        predicted_5sec,
        alert

    ])


    # =====================================================
    # DISPLAY TEXT
    # =====================================================

    cv2.putText(

        frame,

        f"Current Count: {person_count}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (255, 255, 255),

        2

    )

    cv2.putText(

        frame,

        f"Predicted After 5 sec: {predicted_5sec}",

        (20, 80),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        color,

        2

    )

    cv2.putText(

        frame,

        alert,

        (20, 120),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        color,

        2

    )


    # =====================================================
    # WRITE VIDEO
    # =====================================================

    out.write(frame)


    # =====================================================
    # TERMINAL OUTPUT
    # =====================================================

    print(

        f"Frame: {frame_num} | "
        f"Current: {person_count} | "
        f"Predicted: {predicted_5sec}"

    )


    frame_num += 1


# =========================================================
# RELEASE
# =========================================================

cap.release()

out.release()

csv_file.close()

cv2.destroyAllWindows()


# =========================================================
# DONE
# =========================================================

print("\nYOLO Crowd Counting Completed")

print("\nOutput Video:")
print(output_video)

print("\nCSV File:")
print(csv_path)