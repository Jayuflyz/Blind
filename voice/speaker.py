import queue
import threading

import pyttsx3


class VoiceSpeaker:
    """A thread-safe text-to-speech class that speaks from a queue."""

    def __init__(self):
        """Initializes the TTS engine and the speaking thread."""
        self._engine = pyttsx3.init()
        self._speak_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()
        print("✅ VoiceSpeaker initialized.")

    def _process_queue(self):
        """Internal method to process the speech queue."""
        while not self._stop_event.is_set():
            try:
                # Wait for an item to appear in the queue
                text = self._speak_queue.get(timeout=1)
                self._engine.say(text)
                self._engine.runAndWait()
                self._speak_queue.task_done()
            except queue.Empty:
                # This is expected when there's nothing to say
                continue

    def speak(self, text):
        """Adds text to the speech queue to be spoken asynchronously."""
        if text:
            self._speak_queue.put(text)

    def stop(self):
        """Stops the speaker thread and cleans up."""
        print("🛑 Stopping speaker...")
        self._stop_event.set()
        # Clear the queue to ensure the thread can exit if it's waiting
        while not self._speak_queue.empty():
            try:
                self._speak_queue.get_nowait()
            except queue.Empty:
                continue
        self._thread.join()
        print("✅ Speaker stopped.")
