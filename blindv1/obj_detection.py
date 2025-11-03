# import cv2
# import torch
# import pyttsx3
# import numpy as np

# # Initialize TTS engine
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)  # Speed
# engine.setProperty('volume', 1)  # Volume max

# # Load YOLOv5 model (medium for better accuracy)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)
# model.eval()


# # Open webcam (1 or 0 depending on your system)
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# # Resize scale
# scale_percent = 90  # Adjust for balance of speed + accuracy

# spoken_objects = set()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Resize frame
#     width = int(frame.shape[1] * scale_percent / 100)
#     height = int(frame.shape[0] * scale_percent / 100)
#     resized_frame = cv2.resize(frame, (width, height))

#     # Detect objects (disable gradient tracking)
#     with torch.no_grad():
#         results = model(resized_frame[..., ::-1])  # BGR to RGB

#     detections = results.pred[0]
#     labels = results.names
#     detections = detections[detections[:, 4] > 0.45]  # confidence threshold

#     frame_out = results.ims[0]  # rendered image

#     detected_now = []

#     for *box, conf, cls in detections:
#         label = labels[int(cls)]
#         detected_now.append(label)

#         # Draw label manually (to fix readonly image issue)
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 255), 2)
#         cv2.putText(frame_out, f'{label} {conf:.2f}', (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

#     # Voice output for new objects only
#     unique_objs = set(detected_now)
#     new_objects = unique_objs - spoken_objects
#     if new_objects:
#         text = ', '.join(new_objects)
#         engine.say(text)
#         engine.runAndWait()
#         spoken_objects.update(new_objects)

#     # Display frame
#     cv2.imshow('Optimized YOLOv5 Detection', frame_out)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import queue
import threading

import cv2
import pyttsx3
import torch

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1)

# Voice queue and thread
voice_queue = queue.Queue()


def voice_worker():
    while True:
        text = voice_queue.get()
        if text:
            engine.say(text)
            engine.runAndWait()


threading.Thread(target=voice_worker, daemon=True).start()

# Load YOLOv5
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# Open webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Resize scale
scale_percent = 90

spoken_objects = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame, (width, height))

    # Detect objects
    with torch.no_grad():
        results = model(resized_frame[..., ::-1])

    detections = results.pred[0]
    labels = results.names
    detections = detections[detections[:, 4] > 0.45]

    frame_out = resized_frame.copy()
    detected_now = []

    for *box, conf, cls in detections:
        label = labels[int(cls)]
        detected_now.append(label)

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame_out, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Voice output for new detections
    unique_objs = set(detected_now)
    new_objects = unique_objs - spoken_objects
    if new_objects:
        text = ", ".join(new_objects)
        if voice_queue.empty():
            voice_queue.put(text)
        spoken_objects.update(new_objects)

    # Display
    cv2.imshow("Accurate YOLOv5 Detection", frame_out)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
