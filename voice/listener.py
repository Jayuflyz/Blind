import speech_recognition as sr
import threading
import time

class VoiceListener:
    """Listens for voice commands and responds using a VoiceSpeaker instance."""

    def __init__(self, speaker, get_detected_labels_func, hotwords=None, frame_size_func=None):
        """
        Args:
            speaker: VoiceSpeaker instance for speaking responses.
            get_detected_labels_func: Function returning current detected labels.
            hotwords: List of hotwords to trigger listening (default: ["what", "where"]).
            frame_size_func: Function returning (frame_width, frame_height) for relative positions.
        """
        self.speaker = speaker
        self.get_labels = get_detected_labels_func
        self.hotwords = [hw.lower() for hw in (hotwords or ["what", "where"])]
        self.frame_size_func = frame_size_func
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        print("✅ VoiceListener initialized.")

    def start(self):
        """Starts the listening thread."""
        self._thread.start()
        print("🎤 VoiceListener thread started.")

    def stop(self):
        """Stops the listening thread."""
        print("🛑 Stopping listener...")
        self._stop_event.set()
        self._thread.join()
        print("✅ Listener stopped.")

    def _listen_loop(self):
        """Internal loop that listens for voice commands and responds."""
        with self.mic as source:
            print("Calibrating microphone for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        while not self._stop_event.is_set():
            try:
                with self.mic as source:
                    print("🎤 Listening for commands...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=4)

                command = self.recognizer.recognize_google(audio).lower()
                print(f"👂 Detected command: '{command}'")

                if not any(hw in command for hw in self.hotwords):
                    continue

                labels_data = self.get_labels()
                if not labels_data:
                    self.speaker.speak("I don't see any objects right now.")
                    continue

                if "where" in command:
                    self._speak_positions(command, labels_data)
                else:
                    self.speaker.speak("I see " + ", ".join(labels_data.keys()))

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Could not request results from Google service; {e}")
                self.speaker.speak("Sorry, I'm having network issues.")
                time.sleep(2)

    def _speak_positions(self, command, labels):
        """Speaks relative positions of objects."""
        if not self.frame_size_func or not isinstance(labels, dict):
             self.speaker.speak("Position data is not available.")
             return

        frame_width, _ = self.frame_size_func()
        if frame_width == 0:
            return # Avoid division by zero if frame size not ready

        for label, box in labels.items():
            # Check if the user asked for a specific object
            if label in command or len(labels) == 1:
                x1, _, x2, _ = box
                box_center = (x1 + x2) / 2
                if box_center < frame_width / 3:
                    position = "on your left"
                elif box_center < 2 * frame_width / 3:
                    position = "in front of you"
                else:
                    position = "on your right"
                self.speaker.speak(f"The {label} is {position}")

