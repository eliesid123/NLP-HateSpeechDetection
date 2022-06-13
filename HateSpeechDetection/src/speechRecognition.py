from concurrent.futures import thread
import string
import os
from sklearn.metrics import adjusted_mutual_info_score
import speech_recognition as sr

class SpeechManagerResponse:
    error = None
    success = False
    text = string.whitespace

class SpeechManager:
    def __init__(self,mic=None,adjustDuration = 1 ,pauseDuration = 1) -> None:
        self.Microphone = mic
        self.AdjustNoiseDuration = adjustDuration
        self.Recognizer = sr.Recognizer()
        self.Recognizer.pause_threshold = pauseDuration

    def _speechToText(self,audioFile):
        response = SpeechManagerResponse()
        try:
            response.text = self.Recognizer.recognize_google(audioFile, show_all=False)
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
        # except Exception as ex:
            # print(ex)
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
            # print("waiting for audio")
            audioFile = self.Recognizer.listen(mic)
            return self._speechToText(audioFile)

    def Test(self,dir="/home/elie/Desktop/NLP/HateSpeechDetection/datasets/audioTest"):
        for file in os.listdir(dir):
            audio = sr.AudioFile(dir+"/"+file)
            response = self.GetTextFromAudio(audio)
            if response.error == None:
                print(response.text)