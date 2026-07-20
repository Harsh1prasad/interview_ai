try:
    import pyttsx3
    _PYTTSX3_IMPORTED = True
except ImportError as e:
    pyttsx3 = None
    _PYTTSX3_IMPORTED = False
    print(f"[tts] pyttsx3 not installed: {e}")

class TextToSpeech:

    def __init__(self):
        self.available = False
        self.engine = None

        if not _PYTTSX3_IMPORTED:
            return

        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 170)
            self.engine.setProperty("volume", 1.0)
            self.available = True
        except Exception as e:
            # e.g. OSError: no audio driver found on this system
            print(f"[tts] Could not initialize TTS engine: {e}")
            self.available = False

    def speak(self, text):
        if not self.available:
            raise RuntimeError(
                "Text-to-speech isn't available in this environment "
                "(no audio output engine found). Read the question text instead."
            )
        self.engine.say(text)
        self.engine.runAndWait()