import pyttsx3


class TextToSpeech:

    def __init__(self):

        self.engine = pyttsx3.init()

        self.engine.setProperty("rate", 170)

        self.engine.setProperty("volume", 1.0)

    def speak(self, text):

        self.engine.say(text)

        self.engine.runAndWait()