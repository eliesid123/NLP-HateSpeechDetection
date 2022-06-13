import os
import sys

from torch import double
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from modelPrediction import PredictionModel
from speechRecognition import SpeechManager
import speech_recognition as sr
from pynput import keyboard
from colorama import Fore, Back, Style, init, deinit

isPressed = False
Stop = False
isCustomer = False

###########DEFINITIONS####################
modelDir = "/home/elie/Desktop/NLP/HateSpeechDetection/models/"
modelJSONFile = modelDir +  "modelJSON.json"
modelWeightsFile = modelDir + "modelWEIGHTs.h5"
cleanDataPath = "/home/elie/Desktop/NLP/HateSpeechDetection/datasets/cleanData.json"
##########################################

#Adjust those params to have batter results
#################PARAMETERS###################
predictionThreshold = 0.65
pauseThreshhold = 0.5
adjustementDuration = 0.6
##############################################

###################OBJECTS####################
mic = sr.Microphone()
SpeechHandler = SpeechManager(mic,adjustementDuration,pauseThreshhold)
Predictor = PredictionModel(modelJSONFile,modelWeightsFile,cleanDataPath)
##############################################

def onPress(key):
    global isPressed
    global Stop
    global isCustomer
    try: 
        if key == keyboard.Key.space:
            res = SpeechHandler.GetTextFromMic()
            if res.success:
                toPrint = "\nAGENT: " + res.text
                if isCustomer:
                    toPrint = "\nCUSTOMER: " + res.text
                print(Fore.WHITE, Back.LIGHTBLACK_EX, toPrint)
                HandlePrediction(Predictor.Predict(res.text))    
            else:
                print(Fore.LIGHTYELLOW_EX, Back.LIGHTBLACK_EX,"Sorry couldn't here that.")

        elif key == keyboard.Key.esc:
            Stop = True
    except Exception as ex:
        print(ex)
        deinit()
        return

def onRelease(key):
    global isCustomer
    if key == keyboard.Key.space:
        isCustomer = not isCustomer

def PrintHeader():
    #Change Terminal Layout here
    os.system("clear")
    init()
    print(Fore.LIGHTBLUE_EX,Back.LIGHTWHITE_EX,"-----------------Hate Speech Detection App-----------------")
    print("press Space to start recording, ESC to exit.")
    deinit()

def HandlePrediction(prediction):
    init()
    if max(prediction) > predictionThreshold: 
        print(Fore.GREEN,Back.LIGHTBLACK_EX, "NORMAL ({0})".format(max(prediction)))
    else:
        print(Fore.RED,Back.LIGHTBLACK_EX,"OFFENSIVE ({0})".format(max(prediction)))
    
def main():
    PrintHeader()
    with keyboard.Listener(on_press=onPress,on_release=onRelease) as listner:
        listner.join()
    
if __name__ == "__main__":
    main()