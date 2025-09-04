# import torch
# import cv2

# # Load the YOLOv5s pre-trained model from PyTorch Hub
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)

# # Open webcam (0 is default webcam)
# cap = cv2.VideoCapture(1)

# scale_percent = 50

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     width = int(frame.shape[1] * scale_percent / 100)
#     height = int(frame.shape[0] * scale_percent / 100)
#     frame_resize = cv2.resize(frame,(width,height))

#     # Run object detection
#     results = model(frame_resize)

#     # Render results on the frame
#     results.render()

#     # Show the frame with detections
#     cv2.imshow('YOLOv5 Detection', results.ims[0])

#     # Quit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import torch
# import cv2
# import pyttsx3
# import threading

# # Initialize TTS engine
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)  # Speed
# engine.setProperty('volume', 1)  # Volume (0-1)

# # Load YOLOv5s model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)

# # Open webcam
# cap = cv2.VideoCapture(1)

# # Resize scale (to speed up detection)
# scale_percent = 50

# # Keep track of last spoken objects to avoid repetition
# last_spoken = set()

# def speak(text):
#     engine.say(text)
#     engine.runAndWait()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Resize frame to improve speed
#     width = int(frame.shape[1] * scale_percent / 100)
#     height = int(frame.shape[0] * scale_percent / 100)
#     frame_resize = cv2.resize(frame, (width, height))

#     # Detect objects
#     results = model(frame_resize)
#     results.render()

#     # Make copy of output image (to avoid read-only error)
#     output = results.ims[0].copy()

#     # Get detected class labels
#     labels = results.names
#     detected_classes = results.pred[0][:, -1].cpu().numpy() if results.pred[0].shape[0] > 0 else []
#     detected_objects = [labels[int(cls)] for cls in detected_classes]

#     # Speak detected objects (only unique and not repeating every frame)
#     unique_objects = set(detected_objects)
#     new_objects = unique_objects - last_spoken

#     if new_objects:
#         text = ', '.join(new_objects)
#         t = threading.Thread(target=speak, args=(text,))
#         t.start()
#         last_spoken = unique_objects

#     # Show the frame
#     cv2.imshow('YOLOv5 Detection with Voice', output)

#     # Break on 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
