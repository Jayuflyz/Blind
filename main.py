
# # main.py
# from detection.detector import Detector
# from voice.speaker import VoiceSpeaker
# import cv2
# import numpy as np

# # Initialize the detector and speaker
# detector = Detector()
# speaker = VoiceSpeaker()
# spoken_objects = set()

# # Function for Canny edge detection
# def canny_edge_detection(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 100, 200)
#     return edges

# # Function for Harris corner detection (returns binary mask)
# def harris_corner_mask(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = np.float32(gray)
#     dst = cv2.cornerHarris(gray, 2, 3, 0.04)
#     dst = cv2.dilate(dst, None)
#     mask = (dst > 0.01 * dst.max()).astype(np.uint8) * 255
#     return mask

# # Start video capture (try both 0 and 1 if necessary)
# cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
# if not cap.isOpened():
#     cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         print("Failed to grab frame")
#         continue

#     # Get edge and corner masks
#     edges = canny_edge_detection(frame)
#     corners = harris_corner_mask(frame)
#     combined_mask = cv2.bitwise_or(edges, corners)

#     # Create red overlay where mask is active
#     overlay = frame.copy()
#     overlay[combined_mask > 0] = [0, 0, 255]
#     frame_overlayed = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

#     # Detect and draw objects
#     frame_out, new_objects = detector.detect_and_draw(frame_overlayed)

#     # Speak out newly detected objects
#     if new_objects:
#         new_to_speak = new_objects - spoken_objects
#         if new_to_speak:
#             speaker.speak(', '.join(new_to_speak))
#             spoken_objects.update(new_to_speak)

#     # Show final output
#     cv2.imshow("YOLOv5 Detection", frame_out)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import sys
# import time
# # Assuming you have a 'detection' and 'voice' folder in your project
# from detection.detector import Detector
# from voice.speaker import VoiceSpeaker

# # --- Mock/Placeholder classes if you don't have the real files yet ---
# # You can remove this section if you have the actual modules.
# class MockVoiceSpeaker:
#     def speak(self, text):
#         print(f"[SPEAKING]: {text}")

# # If the import fails, use the mock class for testing
# try:
#     from voice.speaker import VoiceSpeaker
# except ImportError:
#     print("Warning: VoiceSpeaker not found. Using mock speaker.")
#     VoiceSpeaker = MockVoiceSpeaker
# # --- End of Mock/Placeholder section ---


# # Initialize modules
# detector = Detector()
# speaker = VoiceSpeaker()

# # Keep short-term memory of spoken objects to avoid spam
# spoken_objects = {}
# SPEAK_TIMEOUT = 10  # seconds before re-announcing the same object

# # ---- Preprocessing Functions ----
# def canny_edge_detection(frame):
#     """Applies Canny edge detection to a frame."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 100, 200)
#     return edges

# def harris_corner_mask(image):
#     """Creates a mask of corners detected by the Harris Corner algorithm."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = np.float32(gray)
#     dst = cv2.cornerHarris(gray, 2, 3, 0.04)
#     dst = cv2.dilate(dst, None)
#     # Create a binary mask where corners are detected
#     mask = (dst > 0.01 * dst.max()).astype(np.uint8) * 255
#     return mask

# # ---- Camera Setup ----
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# if not cap.isOpened():
#     print("❌ Error: Could not open camera")
#     sys.exit(1)

# print("✅ Camera started. Press 'q' to quit.")

# # ---- Main Application Loop ----
# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("Warning: Failed to grab frame from camera.")
#             continue

#         # 1) Run the main detector on the raw camera frame.
#         # It will handle resizing and return a frame ready for display.
#         processed_frame, detected_labels = detector.detect_and_draw(frame)

#         # 2) Compute edge and corner masks on the original, full-resolution frame.
#         edges = canny_edge_detection(frame)
#         corners = harris_corner_mask(frame)
#         combined_mask = cv2.bitwise_or(edges, corners)

#         # 3) Apply the visual overlay to the processed_frame for a consistent size.
#         overlay = processed_frame.copy()
#         # Highlight edges and corners in red on the overlay
#         overlay[combined_mask > 0] = [0, 0, 255] # BGR for red
#         # Blend the overlay with the processed frame
#         final_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
#         # 4) Handle the speech output using a timeout to prevent repetition.
#         current_time = time.time()
#         for label in detected_labels:
#             last_spoken_time = spoken_objects.get(label, 0)
#             if current_time - last_spoken_time > SPEAK_TIMEOUT:
#                 speaker.speak(label)
#                 spoken_objects[label] = current_time

#         # 5) Display the final composed frame to the user.
#         cv2.imshow("Blind Assistant - Detection", final_frame)

#         # Exit loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

# except KeyboardInterrupt:
#     print("🛑 Interrupted by user")

# finally:
#     # Clean up resources
#     cap.release()
#     cv2.destroyAllWindows()
#     print("✅ Resources released, exiting...")


# import cv2
# import numpy as np
# import time
# from detection.detector import Detector
# from voice.speaker import VoiceSpeaker
# from threading import Thread
# import speech_recognition as sr

# # -------------------------
# # Speech Recognition Module
# # -------------------------
# class VoiceListener:
#     def __init__(self, speaker, get_detected_labels, hotword="what"):
#         """
#         speaker: VoiceSpeaker instance
#         get_detected_labels: function returning current YOLO labels
#         hotword: keyword to trigger object info
#         """
#         self.speaker = speaker
#         self.get_detected_labels = get_detected_labels
#         self.hotword = hotword.lower()
#         self.recognizer = sr.Recognizer()
#         self.mic = sr.Microphone()
#         self.listening = True
#         Thread(target=self._listen_loop, daemon=True).start()

#     def _listen_loop(self):
#         while self.listening:
#             try:
#                 with self.mic as source:
#                     self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
#                     print("🎤 Listening for command...")
#                     audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=4)
#                 command = self.recognizer.recognize_google(audio).lower()
#                 print("Command detected:", command)
#                 if self.hotword in command:
#                     labels = self.get_detected_labels()
#                     if labels:
#                         for label in labels:
#                             self.speaker.speak(label)
#                     else:
#                         self.speaker.speak("No objects detected")
#             except sr.WaitTimeoutError:
#                 continue
#             except sr.UnknownValueError:
#                 continue
#             except sr.RequestError:
#                 self.speaker.speak("Network error")
#                 time.sleep(1)

# # -------------------------
# # Helper Functions
# # -------------------------
# def canny_edge_detection(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 100, 200)
#     return edges

# def harris_corner_mask(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = np.float32(gray)
#     dst = cv2.cornerHarris(gray, 2, 3, 0.04)
#     dst = cv2.dilate(dst, None)
#     mask = (dst > 0.01 * dst.max()).astype(np.uint8) * 255
#     return mask

# # -------------------------
# # Initialization
# # -------------------------
# detector = Detector()
# speaker = VoiceSpeaker()
# detected_labels = set()  # Updated each frame

# def get_labels():
#     return detected_labels

# # Start voice listener
# listener = VoiceListener(speaker, get_labels, hotword="what")

# # Video capture
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# if not cap.isOpened():
#     cap = cv2.VideoCapture(1)

# cv2.namedWindow("Blind Assistant - Detection", cv2.WINDOW_NORMAL)

# SPEAK_TIMEOUT = 10
# spoken_objects = {}

# # -------------------------
# # Main Loop
# # -------------------------
# while True:
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         continue

#     # 1) YOLO Detection
#     processed_frame, detected_labels = detector.detect_and_draw(frame)

#     # 2) Overlay (edges + corners)
#     edges = canny_edge_detection(frame)
#     corners = harris_corner_mask(frame)
#     combined_mask = cv2.bitwise_or(edges, corners)
#     overlay = processed_frame.copy()
#     overlay[combined_mask > 0] = [0, 0, 255]
#     final_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)

#     # 3) Automatic speech for newly detected objects
#     current_time = time.time()
#     for label in detected_labels:
#         last = spoken_objects.get(label, 0)
#         if current_time - last > SPEAK_TIMEOUT:
#             speaker.speak(label)
#             spoken_objects[label] = current_time

#     # 4) Display
#     cv2.imshow("Blind Assistant - Detection", final_frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # -------------------------
# # Cleanup
# # -------------------------
# cap.release()
# cv2.destroyAllWindows()
# speaker.stop()
# listener.listening = False



# import cv2
# import numpy as np
# import sys
# import time
# from threading import Lock

# # --- Project Structure Assumption ---
# # your_project/
# # |-- main.py
# # |-- detection/
# # |   |-- detector.py (MUST return a dict of {label: [box]})
# # |-- voice/
# # |   |-- speaker.py
# # |   |-- listener.py (The new advanced version)
# # ------------------------------------

# # --- Mocks for testing if modules are missing ---
# try:
#     from detection.detector import Detector
# except ImportError:
#     print("Warning: Detector not found. Using a mock class for testing.")
#     class Detector:
#         # This mock now returns a dictionary as required by the new listener
#         def detect_and_draw(self, frame): return frame, {'mock_object': [10, 10, 50, 50]}
#         def cleanup(self): pass

# try:
#     from voice.speaker import VoiceSpeaker
# except ImportError:
#     print("Warning: VoiceSpeaker not found. Using a mock class for testing.")
#     class VoiceSpeaker:
#         def speak(self, text): print(f"[SPEAKING]: {text}")
#         def stop(self): pass
        
# try:
#     from voice.listener import VoiceListener
# except ImportError:
#     print("Warning: VoiceListener not found. Using a mock class for testing.")
#     class VoiceListener:
#         def __init__(self, speaker, get_detected_labels_func, hotwords=None, frame_size_func=None): pass
#         def start(self): pass
#         def stop(self): pass
# # --- End Mocks ---

# # ---- Global State & Configuration ----
# # This now stores a dictionary: {label: [box]}
# current_detected_objects = {}
# # This stores the frame size for the listener: (width, height)
# current_frame_size = (0, 0) 
# labels_lock = Lock()
# SPEAK_TIMEOUT = 10 

# # ---- Thread-safe Functions for Listener ----
# def get_current_objects():
#     """Thread-safe function for the listener to get current objects and their boxes."""
#     with labels_lock:
#         return current_detected_objects.copy()

# # Provide a function returning frame size
# def get_frame_size():
#     return processed_frame.shape[1], processed_frame.shape[0]

# # ---- Preprocessing Functions ----
# def canny_edge_detection(frame):
#     """Applies Canny edge detection."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 100, 200)
#     return edges

# def harris_corner_mask(image):
#     """Creates a mask of corners."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = np.float32(gray)
#     dst = cv2.cornerHarris(gray, 2, 3, 0.04)
#     dst = cv2.dilate(dst, None)
#     mask = (dst > 0.01 * dst.max()).astype(np.uint8) * 255
#     return mask

# # ---- Initialization ----
# print("🚀 Initializing system components...")
# detector = Detector()
# speaker = VoiceSpeaker()

# # Initialize the new VoiceListener with multi-hotword and frame size support
# listener = VoiceListener(
#     speaker=speaker,
#     get_detected_labels_func=get_current_objects,
#     hotwords=["what", "where"],
#     frame_size_func=get_frame_size
# )
# listener.start()

# spoken_objects = {} # For auto-announcements

# # ---- Camera Setup ----
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# if not cap.isOpened():
#     print("❌ Error: Could not open camera. Please check the camera index.")
#     sys.exit(1)

# print("✅ Camera started successfully. Press 'q' to quit.")

# # ---- Main Application Loop ----
# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("Warning: Failed to grab frame from camera.")
#             time.sleep(0.5)
#             continue

#         # 1) Run object detection. Expects a dictionary of {label: [box]}.
#         processed_frame, detected_objects = detector.detect_and_draw(frame)
        
#         # 2) Update shared state for the listener thread (thread-safe)
#         with labels_lock:
#             current_detected_objects = detected_objects
#             current_frame_size = (processed_frame.shape[1], processed_frame.shape[0])

#         # 3) Compute and apply visual overlays
#         edges = canny_edge_detection(frame)
#         corners = harris_corner_mask(frame)
#         combined_mask = cv2.bitwise_or(edges, corners)
#         resized_mask = cv2.resize(combined_mask, (processed_frame.shape[1], processed_frame.shape[0]))
        
#         overlay = processed_frame.copy()
#         overlay[resized_mask > 0] = [0, 0, 255] # BGR for red
#         final_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)

#         # 4) Handle automatic speech announcements
#         current_time = time.time()
#         # Iterate over the keys (labels) of the detected objects dictionary
#         for label in detected_objects.keys():
#             last_spoken_time = spoken_objects.get(label, 0)
#             if current_time - last_spoken_time > SPEAK_TIMEOUT:
#                 speaker.speak(label)
#                 spoken_objects[label] = current_time

#         # 5) Display the final frame
#         cv2.imshow("Blind Assistant - Detection", final_frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

# except KeyboardInterrupt:
#     print("\n🛑 Interrupted by user")

# finally:
#     # ---- Cleanup ----
#     print("🧹 Cleaning up resources before shutdown...")
#     listener.stop()
#     speaker.stop()
#     detector.cleanup()
#     cap.release()
#     cv2.destroyAllWindows()
#     print("✅ System shut down gracefully.")


import cv2
import numpy as np
import sys
import time
from threading import Lock

# --- Project Structure Assumption ---
# your_project/
# |-- main.py
# |-- detection/
# |   |-- detector.py (returns dict {label: [box]})
# |-- voice/
# |   |-- speaker.py
# |   |-- listener.py (advanced version with hotwords)
# ------------------------------------

# --- Imports with fallback mocks ---
try:
    from detection.detector import Detector
except ImportError:
    print("Warning: Detector not found. Using mock.")
    class Detector:
        def detect_and_draw(self, frame): return frame, {'mock_object': [10, 10, 50, 50]}
        def cleanup(self): pass

try:
    from voice.speaker import VoiceSpeaker
except ImportError:
    print("Warning: VoiceSpeaker not found. Using mock.")
    class VoiceSpeaker:
        def speak(self, text): print(f"[SPEAKING]: {text}")
        def stop(self): pass

try:
    from voice.listener import VoiceListener
except ImportError:
    print("Warning: VoiceListener not found. Using mock.")
    class VoiceListener:
        def __init__(self, speaker, get_detected_labels_func, hotwords=None, frame_size_func=None): pass
        def start(self): pass
        def stop(self): pass
# --- End mocks ---

# ---- Global State & Configuration ----
current_detected_objects = {}  # {label: [box]}
current_frame_size = (0, 0)    # (width, height)
labels_lock = Lock()
SPEAK_TIMEOUT = 10  # seconds before repeating auto-speak

# ---- Thread-safe getters for listener ----
def get_current_objects():
    with labels_lock:
        return current_detected_objects.copy()

def get_frame_size():
    with labels_lock:
        return current_frame_size

# ---- Preprocessing functions ----
def canny_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, 100, 200)

def harris_corner_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    mask = (dst > 0.01 * dst.max()).astype(np.uint8) * 255
    return mask

# ---- Initialization ----
print("🚀 Initializing system components...")
detector = Detector()
speaker = VoiceSpeaker()

listener = VoiceListener(
    speaker=speaker,
    get_detected_labels_func=get_current_objects,
    hotwords=["what", "where"],
    frame_size_func=get_frame_size
)
listener.start()

spoken_objects = {}  # tracks last auto-announced timestamps

# ---- Camera Setup ----
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    sys.exit(1)
print("✅ Camera started successfully. Press 'q' to quit.")

# ---- Main Loop ----
try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Failed to grab frame.")
            time.sleep(0.5)
            continue

        # 1) Object detection
        processed_frame, detected_objects = detector.detect_and_draw(frame)

        # 2) Update shared state
        with labels_lock:
            current_detected_objects = detected_objects
            current_frame_size = (processed_frame.shape[1], processed_frame.shape[0])

        # 3) Overlay edges and corners
        edges = canny_edge_detection(frame)
        corners = harris_corner_mask(frame)
        combined_mask = cv2.bitwise_or(edges, corners)
        resized_mask = cv2.resize(combined_mask, (processed_frame.shape[1], processed_frame.shape[0]))
        overlay = processed_frame.copy()
        overlay[resized_mask > 0] = [0, 0, 255]  # red overlay
        final_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)

        # 4) Auto speech announcements
        current_time = time.time()
        for label in detected_objects.keys():
            last_spoken_time = spoken_objects.get(label, 0)
            if current_time - last_spoken_time > SPEAK_TIMEOUT:
                speaker.speak(label)
                spoken_objects[label] = current_time

        # 5) Display
        cv2.imshow("Blind Assistant - Detection", final_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user")

finally:
    print("🧹 Cleaning up resources...")
    listener.stop()
    speaker.stop()
    detector.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("✅ System shut down gracefully.")
