"""https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/"""

from dataManagement import DataSetManager
import os
from tokenizeros import Tokenizor
from modelBuilder import CnnModel
from speechRecognition import SpeechManager,SpeechManagerResponse
import speech_recognition as sr

dir = os.getcwd()

def main():
	path = os.path.join("/home","elie","Desktop","NLP","HateSpeechDetection","datasets","training-v1","offenseval-training-v1.tsv")
	tokenizer  = Tokenizor()
	reader = DataSetManager(path,tokenizer,0.7)
	reader.CleanData()
	reader.TokenizeData()
	trainTexts, trainLabels = reader.GetTrainingData()
	testData, testLabels = reader.GetValidationData()
	model = CnnModel()
	model.Create(PRINT=True)
	model.Compile()
	results = model.Train(trainTexts,trainLabels,testData,testLabels)
	print(results)

def test():
	model = CnnModel()
	model.Create(PRINT=True)
	model.Compile()
		
if __name__ == "__main__":
	# main()
	test()