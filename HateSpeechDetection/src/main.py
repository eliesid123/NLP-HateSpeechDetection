"""https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/"""

import os
from tokenizeros import Tokenizor
from dataManagement import DataSetManager
from modelPrediction import PredictionModel
from modelBuilder import CnnModel
from speechRecognition import SpeechManager,SpeechManagerResponse
import speech_recognition as sr

dir = os.getcwd()

# def main():
# 	path = os.path.join("/home/elie/Desktop/NLP/HateSpeechDetection/datasets/training-v1/offenseval-training-v1.tsv")
# 	tokenizer  = Tokenizor()
# 	reader = DataSetManager(path,tokenizer,0.7)
# 	reader.CleanData()
# 	reader.TokenizeData()
# 	trainTexts, trainLabels = reader.GetTrainingData()
# 	testTexts, testLabels = reader.GetValidationData()
# 	model = CnnModel()
# 	model.Create(PRINT=True)
# 	model.Compile()
# 	results = model.Train(trainTexts,trainLabels,testTexts,testLabels)
# 	print(results)

def CreateModel():
	path = "/home/elie/Desktop/NLP/HateSpeechDetection/datasets/training-v1/offenseval-training-v1.tsv"
	cleanDataPath = "/home/elie/Desktop/NLP/HateSpeechDetection/datasets/cleanData.jspn"
	reader = DataSetManager(path,cleanDataPath=cleanDataPath, dataSplit = 0.7)
	reader.CleanData(isSave=True)
	trainTexts, trainLabels = reader.GetTrainingData()
	testTexts, testLabels = reader.GetValidationData()
	allData, all = reader.GetAllData() 
	model = CnnModel()
	model.Create(PRINT=True)
	model.Compile(allData)
	model.Train(trainTexts,trainLabels,testTexts,testLabels)
	model.SaveModel()		

def testPrediction():
	dir = "/home/elie/Desktop/NLP/HateSpeechDetection/"
	cleanDataPath = dir+"datasets/cleanData.json"
	modelDir = dir+"models/"
	modelJSONFile = modelDir +  "modelJSON.json"
	modelWeightsFile = modelDir + "modelWEIGHTs.h5"
	path = dir+"datasets/training-v1/offenseval-training-v1.tsv"
	reader = DataSetManager(path,cleanDataPath=cleanDataPath, dataSplit = 0.7)
	reader.CleanData(isSave=True)
	predictionModel = PredictionModel(modelJSONFile,modelWeightsFile,cleanDataPath)
	
	testTexts, testLabels = reader.GetValidationData()
	predictionModel.Test(testTexts,testLabels)
	predictionModel.Predict("what a stupid man, are you crazy?")

if __name__ == "__main__":
	# main()
	# testPrediction()
	CreateModel()