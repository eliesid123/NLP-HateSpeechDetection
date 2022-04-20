from dataManagement import DataSetManager
import os
from tokenizeros import Tokenizor
from modelBuilder import CnnModel

dir = os.getcwd()

def main():
	path = os.path.join("/home","elie","Desktop","NLP","HateSpeechDetection","datasets","training-v1","offenseval-training-v1.tsv")
	reader = DataSetManager(path,Tokenizor(),0.7)
	reader.CleanData()
	reader.TokenizeData()
	trainTexts, trainLabels = reader.GetTrainingData()
	testData, testLabels = reader.GetValidationData()
	model = CnnModel(reader)
	model.Create(PRINT=True)
	model.Compile()
	results = model.Train(trainTexts,trainLabels,testData,testLabels)
	print(results)

if __name__ == "__main__":
	main()