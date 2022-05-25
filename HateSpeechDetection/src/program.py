from modelPrediction import PredictionModel
from speechRecognition import SpeechManager,SpeechManagerResponse
import speech_recognition as sr
from pynput.keyboard import Listener, Key

###########DEFINITIONS####################
modelDir = "/home/elie/Desktop/NLP/HateSpeechDetection/models/"
modelJSONFile = modelDir +  "modelJSON.json"
modelWeightsFile = modelDir + "modelWEIGHTs.h5"
cleanDataPath = "/home/elie/Desktop/NLP/HateSpeechDetection/datasets/cleanData.json"
##########################################

#get mic
mic = sr.Microphone.get_pyaudio()
SpeechHandler = SpeechManager(mic)
Predictor = PredictionModel(modelJSONFile,modelWeightsFile,cleanDataPath)
isPressed = False
Stop = False


def onPress(key):
    global isPressed
    global Stop
    if key == 'r':
        isPressed = True
    elif key == 's':
        Stop = True

def onRelease(key):
    global isPressed
    if key == 'r':
        isPressed = False
        

def PrintHeader():
    print("NLP App....")

def HandlePrediction(prediction):
    pass

def main():
    isCustomer = False
    PrintHeader()
    with Listener(on_press=onPress,on_release=onRelease) as listner:
        listner.join()
        while not Stop:
            if isPressed:
                res = SpeechHandler.GetTextFromAudio()
                while res.success:
                    toPrint = "AGENT: " + res.text
                    if isCustomer:
                        toPrint = "CUSTOMER: " + res.text
                    print(toPrint)
                    HandlePrediction(Predictor.Predict(res.text))
                    break