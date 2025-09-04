# from .sort import Sort  # we will install this soon
# import numpy as np

# class ObjectTracker:
#     def __init__(self):
#         self.tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)

#     def update(self, detections):
#         # detections: np.array([[x1,y1,x2,y2,score], ...])
#         tracked_objects = self.tracker.update(detections)
#         # tracked_objects: np.array([[x1,y1,x2,y2,track_id], ...])
#         return tracked_objects


import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def get_hand_box_and_keypoints(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        hand_box = None
        keypoints = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                xmin, xmax = int(min(x_coords)), int(max(x_coords))
                ymin, ymax = int(min(y_coords)), int(max(y_coords))
                hand_box = [xmin, ymin, xmax, ymax]

                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    keypoints.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

                # Optionally draw full hand
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return hand_box, keypoints
