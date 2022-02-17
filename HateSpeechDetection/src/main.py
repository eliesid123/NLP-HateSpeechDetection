from asyncore import read
from contextlib import redirect_stderr
import re
from dataManagement import DataSetManager
import os
import tokenize

from modelBuilder import ModelBuilder

def Tokenize(sentence):
    return tokenize.tokenize(sentence)

dir = os.getcwd()

def main():
	path = os.path.join(dir,"datasets","training-v1","offenseval-training-v1.tsv")
	reader = DataSetManager(path,0.9)
	reader.CleanData()
	trainTexts, valTexts, trainLabels, valLabels = reader.GetTrainingData()
	testData, testLabels = reader.GetTestingData()
	model = ModelBuilder()
	model.Compile()
	results = model.Train(trainTexts,trainLabels,valTexts,valLabels)
	print(results)
if __name__ == "__main__":
	main()