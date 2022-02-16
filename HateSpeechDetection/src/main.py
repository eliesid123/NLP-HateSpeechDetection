from asyncore import read
from contextlib import redirect_stderr
import re
from dataManagement import DataSetManager
import os
import tokenize

def Tokenize(sentence):
    return tokenize.tokenize(sentence)

dir = os.getcwd()

def main():
	path = os.path.join(dir,"datasets","training-v1","offenseval-training-v1.tsv")
	reader = DataSetManager(path,0.9)
	reader.CleanData()
	trainTexts, valTexts, trainLabels, valLabels = reader.GetTrainingData()
	testData, testLabels = reader.GetTestingData()
	tokens = Tokenize(trainTexts[1])
	print(tokens)
if __name__ == "__main__":
	main()