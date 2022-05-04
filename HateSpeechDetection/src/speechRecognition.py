import string
import os
import speech_recognition as sr
from torch import set_flush_denormal

class SpeechManagerResponse:
    error = None
    success = False
    text = string.whitespace

class SpeechManager:
    def __init__(self,mic=None) -> None:
        self.Microphone = mic
        self.AdjustNoiseDuration = 1
        self.Recognizer = sr.Recognizer()
        self.Recognizer.pause_threshold = 2

    def _speechToText(self,audioFile):
        response = SpeechManagerResponse()
        try:
            response.text = self.Recognizer.recognize_google(audioFile,show_all=True)
            response.success = True
        except sr.UnknownValueError:
            response.error = "audio file unintelligible"
            response.success = False
        except sr.RequestError:
            try:
                response.text = self.Recognizer.recognize_sphinx(audioFile)
                response.success = True
            except:
                response.error = "Couldn't detect speech"
                response.success = False
        finally:
            return response

    def GetTextFromAudio(self,audio):
        with audio as source:
            self.Recognizer.adjust_for_ambient_noise(source, self.AdjustNoiseDuration)
            audioFile =  self.Recognizer.record(source)
            return self._speechToText(audioFile)

    def GetTextFromMic(self):
        with self.Microphone as mic:
            self.Recognizer.adjust_for_ambient_noise(mic, self.AdjustNoiseDuration)
            print("waiting for audio")
            audioFile = self.Recognizer.listen(mic)
            return self._speechToText(audioFile)

    def Test(self,dir="/home/elie/Desktop/NLP/HateSpeechDetection/datasets/audioTest"):
        for file in os.listdir(dir):
            audio = sr.AudioFile(dir+"/"+file)
            response = self.GetTextFromAudio(audio)
            if response.error == None:
                print(response.text)