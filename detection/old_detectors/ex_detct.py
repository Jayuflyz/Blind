# import torch
# import cv2
# import numpy as np
# from config.settings import CONFIDENCE, SCALE_PERCENT
# from tracking.tracker import ObjectTracker  # 👈 import the tracker


# class Detector:
#     def __init__(self):
#         self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model.to(self.device).eval()
#         self.tracker = ObjectTracker()

# def detect_and_draw(self, frame):
#     width = int(frame.shape[1] * SCALE_PERCENT / 100)
#     height = int(frame.shape[0] * SCALE_PERCENT / 100)
#     resized_frame = cv2.resize(frame, (width, height))

#     with torch.no_grad():
#         results = self.model(resized_frame[..., ::-1])

#     detections = results.pred[0]
#     labels = results.names
#     detections = detections[detections[:, 4] > CONFIDENCE]

#     # Convert to numpy for tracking
#     detections_np = detections.cpu().numpy()
#     detections_for_tracking = detections_np[:, :5]  # [x1, y1, x2, y2, conf]

#     tracked_objects = self.tracker.update(detections_for_tracking)

#     detected_labels = set()

#     # Draw tracked boxes with IDs
#     for *box, track_id in tracked_objects:
#         x1, y1, x2, y2 = map(int, box)
#         label = f'ID {int(track_id)}'
#         cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(resized_frame, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # Optionally still get YOLO class names for TTS
#     for *box, conf, cls in detections:
#         label = labels[int(cls)]
#         detected_labels.add(label)

#     return resized_frame, detected_labels


# import torch
# import cv2
# import numpy as np
# from config.settings import CONFIDENCE, SCALE_PERCENT
# from tracking.tracker import ObjectTracker

# class Detector:
#     def __init__(self):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         # Load YOLOv5 model with autoshape enabled for easy results.pred access
#         self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=True)
#         self.model.to(self.device).eval()

#         # Use half precision only on CUDA devices
#         if self.device == 'cuda':
#             self.model.half()
#         else:
#             self.model.float()

#         self.labels = self.model.names  # class names
#         self.tracker = ObjectTracker()

#     def draw_box(self, img, box, track_id, label):
#         x1, y1, x2, y2 = map(int, box)
#         color = (0, 255, 0)  # green box
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         text = f'{label} ID:{track_id}'
#         cv2.putText(img, text, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     def detect_and_draw(self, frame):
#         width = int(frame.shape[1] * SCALE_PERCENT / 100)
#         height = int(frame.shape[0] * SCALE_PERCENT / 100)
#         resized_frame = cv2.resize(frame, (width, height))

#         img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

#         with torch.no_grad():
#             results = self.model(img_rgb)  # pass raw RGB numpy array

#         detections = results.xyxy[0]  # detections: [x1,y1,x2,y2,conf,cls]
#         detections = detections[detections[:, 4] > CONFIDENCE]

#         detected_labels = set()

#         if detections.shape[0] > 0:
#             boxes_scores = detections[:, :5].cpu().numpy()  # tracker needs [x1, y1, x2, y2, conf]
#             classes = detections[:, 5].cpu().numpy().astype(int)  # store class IDs separately
#             tracked = self.tracker.update(boxes_scores)

#             for i, (*box, track_id) in enumerate(tracked):
#                 cls = int(detections[i, 5].item())
#                 label = self.labels[cls]
#                 detected_labels.add(label)

#                 # Clamp box coords within image
#                 x1, y1, x2, y2 = map(int, box)
#                 x1 = max(0, min(x1, resized_frame.shape[1] - 1))
#                 y1 = max(0, min(y1, resized_frame.shape[0] - 1))
#                 x2 = max(0, min(x2, resized_frame.shape[1] - 1))
#                 y2 = max(0, min(y2, resized_frame.shape[0] - 1))

#                 self.draw_box(resized_frame, (x1, y1, x2, y2), track_id, label)

#         return resized_frame, detected_labels



# import torch
# import cv2
# import numpy as np
# from config.settings import CONFIDENCE, SCALE_PERCENT
# from tracking.tracker import ObjectTracker
# from voice.speaker import VoiceSpeaker  
# from voice.translations import LABEL_TRANSLATIONS

# class Detector:
#     def __init__(self):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=True)
#         self.model.to(self.device).eval()

#         if self.device == 'cuda':
#             self.model.half()
#         else:
#             self.model.float()

#         self.labels = self.model.names
#         self.tracker = ObjectTracker()
#         self.speaker = VoiceSpeaker()  # 🗣️ Initialize your speaker
#         self.previous_labels = set()   # 🧠 Memory of seen labels

#     def draw_box(self, img, box, track_id, label):
#         x1, y1, x2, y2 = map(int, box)
#         color = (0, 255, 0)
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         text = f'{label} ID:{track_id}'
#         cv2.putText(img, text, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     def detect_and_draw(self, frame):
#         width = int(frame.shape[1] * SCALE_PERCENT / 100)
#         height = int(frame.shape[0] * SCALE_PERCENT / 100)
#         resized_frame = cv2.resize(frame, (width, height))

#         img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

#         with torch.no_grad():
#             results = self.model(img_rgb)

#         detections = results.xyxy[0]
#         detections = detections[detections[:, 4] > CONFIDENCE]

#         current_labels = set()

#         if detections.shape[0] > 0:
#             boxes_scores = detections[:, :5].cpu().numpy()
#             classes = detections[:, 5].cpu().numpy().astype(int)
#             tracked = self.tracker.update(boxes_scores)

#             for i, (*box, track_id) in enumerate(tracked):
#                 cls = int(detections[i, 5].item())
#                 original_label = self.labels[cls]
#                 label = LABEL_TRANSLATIONS.get(original_label, original_label)
#                 current_labels.add(label)

#                 x1, y1, x2, y2 = map(int, box)
#                 x1 = max(0, min(x1, resized_frame.shape[1] - 1))
#                 y1 = max(0, min(y1, resized_frame.shape[0] - 1))
#                 x2 = max(0, min(x2, resized_frame.shape[1] - 1))
#                 y2 = max(0, min(y2, resized_frame.shape[0] - 1))

#                 self.draw_box(resized_frame, (x1, y1, x2, y2), track_id, label)

#         # 🧠 Memory comparison: what’s new?
#         new_labels = current_labels - self.previous_labels
#         if new_labels:
#             message = ', '.join(new_labels)
#             self.speaker.speak(message)  # 🗣️ Speak only new things
#             self.previous_labels = current_labels.copy()

#         return resized_frame, current_labels
