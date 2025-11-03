# import cv2
# import numpy as np
# import time
# import torch
# from ultralytics import YOLO
# # Assuming these are in your project structure
# from config.settings import CONFIDENCE, SCALE_PERCENT
# from tracking.tracker import HandTracker
# from voice.speaker import VoiceSpeaker
# from learning.learner import ObjectLearner

# # --- Mock/Placeholder classes for demonstration if the actual files are not available ---
# # If you have these files, you can remove this section.
# class HandTracker:
#     def get_hand_box_and_keypoints(self, frame):
#         # Placeholder: returns no hand detected
#         return None, []

# class VoiceSpeaker:
#     def speak(self, text):
#         print(f"[SPEAKING]: {text}")

# class ObjectLearner:
#     pass # Placeholder

# CONFIDENCE = 0.6
# SCALE_PERCENT = 50
# # --- End of Mock/Placeholder section ---


# class Detector:
#     def __init__(self, config=None):
#         # Default config
#         self.config = {
#             "confidence_threshold": CONFIDENCE,
#             "scale_percent": SCALE_PERCENT,
#             "hand_proximity_threshold": 100,  # pixels
#             "speech_cooldown": 3,  # seconds
#             "known_objects": ["person", "car", "dog", "chair", "bicycle"],
#             "performance_monitoring": True,
#             "frame_skip": 5  # only process 1 out of N frames
#         }
#         if config:
#             self.config.update(config)

#         # Init model + components
#         self._initialize_model()
#         self._initialize_components()

#         # Performance
#         self.frame_count = 0
#         self.start_time = time.time()
#         self.fps = 0

#         # Memory for speaking control
#         self.last_spoken = {}
#         self.recent_objects = {}

#         # Initialize the last display frame to prevent errors on the first run
#         self.last_display_frame = None

#     def _initialize_model(self):
#         """Initialize YOLOv8 model properly"""
#         try:
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#             print(f"Using device: {self.device}")

#             self.model = YOLO("yolov8n.pt")
#             self.model.to(self.device)

#             print(f"Model loaded with {len(self.model.names)} classes")
#         except Exception as e:
#             print(f"Error initializing model: {e}")
#             raise

#     def _initialize_components(self):
#         """Initialize extra components"""
#         try:
#             self.hand_tracker = HandTracker()
#             self.speaker = VoiceSpeaker()
#             self.learner = ObjectLearner()

#             self.hand_label = "hand"
#             self.is_listening = False
#             self.listening_start_time = 0
#         except Exception as e:
#             print(f"Error initializing components: {e}")
#             raise

#     def draw_box(self, img, box, label, color=(0, 255, 0), confidence=None):
#         """Draw bounding box with label and optional confidence"""
#         try:
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

#             label_text = label
#             if confidence is not None:
#                 label_text += f" ({confidence:.2f})"

#             (text_width, text_height), baseline = cv2.getTextSize(
#                 label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
#             )
#             cv2.rectangle(img, (x1, y1 - text_height - 10),
#                           (x1 + text_width, y1), color, -1)
#             cv2.putText(img, label_text, (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         except Exception as e:
#             print(f"Error drawing box: {e}")

#     def _process_detections(self, results):
#         """Filter detections by confidence"""
#         try:
#             detections = []
#             for box in results[0].boxes:
#                 conf = float(box.conf)
#                 cls_id = int(box.cls)
#                 if conf >= self.config["confidence_threshold"]:
#                     label = self.model.names[cls_id]
#                     xyxy = box.xyxy[0].cpu().numpy().astype(int)
#                     detections.append((xyxy, label, conf))
#             return detections
#         except Exception as e:
#             print(f"Error processing detections: {e}")
#             return []

#     def _speak_object(self, label, current_time):
#         """Smart speaking: avoid repeats, handle unknowns gracefully"""
#         last_spoke = self.last_spoken.get(label, 0)
#         if current_time - last_spoke < self.config["speech_cooldown"]:
#             return  # too soon to repeat

#         if label not in self.config["known_objects"]:
#             self.speaker.speak("Obstacle ahead")
#         else:
#             # check if already recently announced
#             if label in self.recent_objects and (current_time - self.recent_objects[label]) < 10:
#                 return
#             self.speaker.speak(label)
#             self.recent_objects[label] = current_time

#         self.last_spoken[label] = current_time

#     def _handle_hand_interaction(self, resized, hand_box, keypoints, other_boxes, current_time):
#         """Process detections near the hand"""
#         detected_labels = set()
#         self.draw_box(resized, hand_box, "Hand", color=(0, 255, 255))

#         if keypoints:
#             for kp in keypoints:
#                 x, y = int(kp[0]), int(kp[1])
#                 cv2.circle(resized, (x, y), 4, (0, 0, 255), -1)

#         for obj_box, obj_label, conf in other_boxes:
#             distance = self.keypoints_to_box_distance(keypoints, obj_box)
#             if distance < self.config["hand_proximity_threshold"]:
#                 self._speak_object(obj_label, current_time)
#                 self.draw_box(resized, obj_box, obj_label, confidence=conf)
#                 detected_labels.add(obj_label)
#         return detected_labels

#     def _handle_regular_detection(self, resized, other_boxes, current_time):
#         """Normal detection without hand"""
#         detected_labels = set()
#         for obj_box, obj_label, conf in other_boxes:
#             self.draw_box(resized, obj_box, obj_label, confidence=conf)
#             self._speak_object(obj_label, current_time)
#             detected_labels.add(obj_label)
#         return detected_labels

#     def _update_performance_metrics(self):
#         """Update FPS"""
#         if not self.config["performance_monitoring"]:
#             return

#         elapsed = time.time() - self.start_time
#         if elapsed > 1:
#             self.fps = self.frame_count / elapsed
#             self.frame_count = 0
#             self.start_time = time.time()

#     def _draw_ui_elements(self, frame):
#         """Draw FPS and status"""
#         if self.config["performance_monitoring"]:
#             cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     def detect_and_draw(self, frame):
#         """Main loop - always return a display frame with the same size as 'frame'."""
#         try:
#             # Frame bookkeeping for skipping logic
#             self.frame_count += 1
#             do_process = (self.frame_count % self.config.get("frame_skip", 1) == 0)

#             # Compute resized version for model processing (preserve original for display)
#             width = int(frame.shape[1] * self.config['scale_percent'] / 100)
#             height = int(frame.shape[0] * self.config['scale_percent'] / 100)
#             resized = cv2.resize(frame, (width, height))
#             img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

#             current_time = time.time()

#             # Light-weight hand tracking can run every frame (optional)
#             hand_box, keypoints = self.hand_tracker.get_hand_box_and_keypoints(resized)

#             detected_labels = set()

#             if do_process:
#                 # Run YOLO only when we decided to process this frame
#                 with torch.no_grad():
#                     results = self.model(img_rgb)

#                 other_boxes = self._process_detections(results)

#                 if hand_box:
#                     detected_labels = self._handle_hand_interaction(
#                         resized, hand_box, keypoints, other_boxes, current_time
#                     )
#                 else:
#                     detected_labels = self._handle_regular_detection(
#                         resized, other_boxes, current_time
#                     )

#                 # Update perf metrics before drawing UI
#                 self._update_performance_metrics()
#                 self._draw_ui_elements(resized)

#                 # Scale processed (resized) frame back up to original camera resolution for display
#                 display_frame = cv2.resize(resized, (frame.shape[1], frame.shape[0]))

#                 # Save as last display frame so skipped frames can reuse it (prevents flicker)
#                 self.last_display_frame = display_frame.copy()

#             else:
#                 # Skipping heavy model run: reuse last display frame if available
#                 # If we have a last processed display frame, use it (preferred)
#                 if self.last_display_frame is not None:
#                     display_frame = self.last_display_frame.copy()
#                     # We can still update the FPS on the copied frame to show it's live
#                     self._draw_ui_elements(display_frame)
#                 else:
#                     # No last frame yet: make a simple display from the original frame
#                     display_frame = frame.copy()
#                     self._draw_ui_elements(display_frame)

#             return display_frame, detected_labels

#         except Exception as e:
#             print(f"Error in detect_and_draw: {e}")
#             # On error return original frame to avoid crashing the video feed
#             return frame, set()

#     def keypoints_to_box_distance(self, keypoints, box):
#         """Distance between hand and object"""
#         try:
#             box_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
#             if keypoints and len(keypoints) >= 9: # Index 8 is the tip of the index finger
#                 fingertip = keypoints[8]
#                 return np.linalg.norm(np.array(fingertip) - np.array(box_center))
#             return float("inf")
#         except Exception as e:
#             print(f"Error calculating distance: {e}")
#             return float("inf")

#     def get_performance_stats(self):
#         return {"fps": self.fps, "device": self.device}

#     def cleanup(self):
#         try:
#             if hasattr(self, "model"):
#                 del self.model
#             if hasattr(self, "hand_tracker"):
#                 del self.hand_tracker
#             print("Resources cleaned up successfully")
#         except Exception as e:
#             print(f"Error during cleanup: {e}")

import time

import cv2
import torch
from ultralytics import YOLO


# Placeholder classes if needed
class VoiceSpeaker:
    def speak(self, text):
        print(f"[SPEAKING]: {text}")


class ObjectLearner:
    pass


CONFIDENCE = 0.6
SCALE_PERCENT = 50


class Detector:
    def __init__(self, config=None):
        self.config = {
            "confidence_threshold": CONFIDENCE,
            "scale_percent": SCALE_PERCENT,
            "speech_cooldown": 5,
            "known_objects": ["person", "car", "dog", "chair", "bicycle", "bottle", "cup"],
            "performance_monitoring": True,
            "frame_skip": 2,  # Process every 2nd frame
        }
        if config:
            self.config.update(config)

        self._initialize_model()
        self._initialize_components()

        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

        self.last_spoken = {}
        self.last_display_frame = None

    def _initialize_model(self):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            self.model = YOLO("yolov8s.pt")
            self.model.to(self.device)
            print(f"Model loaded with {len(self.model.names)} classes")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def _initialize_components(self):
        try:
            self.speaker = VoiceSpeaker()
            self.learner = ObjectLearner()
        except Exception as e:
            print(f"Error initializing components: {e}")
            raise

    def draw_box(self, img, box, label, color=(0, 255, 0), confidence=None):
        try:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label_text = label
            if confidence is not None:
                label_text += f" ({confidence:.2f})"

            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            print(f"Error drawing box: {e}")

    def _process_detections(self, results):
        try:
            detections = []
            for box in results[0].boxes:
                conf = float(box.conf)
                cls_id = int(box.cls)
                if conf >= self.config["confidence_threshold"]:
                    label = self.model.names[cls_id]
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append((xyxy, label, conf))
            return detections
        except Exception as e:
            print(f"Error processing detections: {e}")
            return []

    def _speak_object(self, label, current_time):
        last_spoke = self.last_spoken.get(label, 0)
        if current_time - last_spoke < self.config["speech_cooldown"]:
            return
        if label in self.config["known_objects"]:
            self.speaker.speak(label)
            self.last_spoken[label] = current_time

    def _handle_detection(self, resized, detections, current_time):
        detected_objects = {}
        for obj_box, obj_label, conf in detections:
            self.draw_box(resized, obj_box, obj_label, confidence=conf)
            self._speak_object(obj_label, current_time)
            detected_objects[obj_label] = obj_box
        return detected_objects

    def _update_performance_metrics(self):
        if not self.config["performance_monitoring"]:
            return
        elapsed = time.time() - self.start_time
        if elapsed > 1:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    def _draw_ui_elements(self, frame):
        if self.config["performance_monitoring"]:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def detect_and_draw(self, frame):
        try:
            self.frame_count += 1
            do_process = self.frame_count % self.config.get("frame_skip", 1) == 0

            if do_process:
                width = int(frame.shape[1] * self.config["scale_percent"] / 100)
                height = int(frame.shape[0] * self.config["scale_percent"] / 100)
                resized = cv2.resize(frame, (width, height))
                img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                with torch.no_grad():
                    results = self.model(img_rgb)

                detections = self._process_detections(results)
                detected_objects = self._handle_detection(resized, detections, time.time())

                self._update_performance_metrics()
                self._draw_ui_elements(resized)

                display_frame = cv2.resize(resized, (frame.shape[1], frame.shape[0]))
                self.last_display_frame = display_frame.copy()

            else:
                detected_objects = {}
                if self.last_display_frame is not None:
                    display_frame = self.last_display_frame.copy()
                    self._draw_ui_elements(display_frame)
                else:
                    display_frame = frame.copy()
                    self._draw_ui_elements(display_frame)

            return display_frame, detected_objects

        except Exception as e:
            print(f"Error in detect_and_draw: {e}")
            return frame, {}

    def get_performance_stats(self):
        return {"fps": self.fps, "device": self.device}

    def cleanup(self):
        try:
            if hasattr(self, "model"):
                del self.model
            print("Resources cleaned up successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")
