# learning/learner.py
import os
import cv2
import uuid
import speech_recognition as sr

class ObjectLearner:
    def __init__(self, save_dir='data/unknown_objects'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.recognizer = sr.Recognizer()

    def ask_and_get_label(self):
        try:
            with sr.Microphone() as source:
                print("🎤 Listening for object name...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                label = self.recognizer.recognize_google(audio)
                print(f"✅ Heard label: {label}")
                return label.strip().lower()
        except sr.UnknownValueError:
            print("❌ Could not understand speech.")
            return None
        except sr.RequestError as e:
            print(f"🔌 Speech service error: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Error: {e}")
            return None

    def save_cropped_object(self, frame, box, label):
        x1, y1, x2, y2 = map(int, box)
        cropped = frame[y1:y2, x1:x2]
        label_dir = os.path.join(self.save_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        filename = os.path.join(label_dir, f"{uuid.uuid4().hex}.jpg")
        cv2.imwrite(filename, cropped)
        print(f"💾 Saved object as: {filename}")
