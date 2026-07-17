
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHANNELS = 1


class SpeechToText:
    import sounddevice as sd
    import soundfile as sf

    def __init__(self, model_size="base"):
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )

    def record_audio(self, duration=10, filename="answer.wav"):

        print("Recording...")

        recording = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16"
        )

        sd.wait()

        sf.write(filename, recording, SAMPLE_RATE)

        print("Recording Finished")

        return filename

    def transcribe(self, audio_path):

        segments, _ = self.model.transcribe(audio_path)

        text = ""

        for segment in segments:
            text += segment.text + " "

        return text.strip()