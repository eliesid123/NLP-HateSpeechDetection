import string
import struct
from turtle import pensize
from typing import final
from urllib import response
import speech_recognition as sr

class SpeechManagerResponse:
    error = None
    success = False
    text = string.whitespace

class SpeechManager:
    def __init__(self,mic) -> None:
        self.Microphone = mic
        self.AdjustNoiseDuration = 1
        self.Recognizer = sr.Recognizer()

    def _speechToText(self,audioFile):
        response = SpeechManagerResponse()
        with audioFile as source:
            try:
                response.text = self.Recognizer.recognize_google(source)
                response.success = True
            except sr.UnknownValueError:
                response.error = "audio file unintelligible"
                response.success = False
            except sr.RequestError:
                try:
                    response.text = self.Recognizer.recognize_sphinx(source)
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
            audioFile = self.Recognizer.listen(mic)
            return self._speechToText(audioFile)