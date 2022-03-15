from dataManagement import DataSetManager
import os
from tokenizeros import Tokenizor
from modelBuilder import ModelBuilder

dir = os.getcwd()

def main():
	path = os.path.join("home","elie","Desktop","NLP","HateSpeechDetection","datasets","training-v1","datasets","training-v1","offenseval-training-v1.tsv")
	reader = DataSetManager(path,0.1)
	reader.CleanData()
	trainTexts, valTexts, trainLabels, valLabels = reader.GetTrainingData()
	testData, testLabels = reader.GetTestingData()
	tokenizer = Tokenizor()
	model = ModelBuilder(tokenizer)
	model.Create(PRINT=True)
	model.Compile()
	results = model.Train(trainTexts,trainLabels)
	print(results)
if __name__ == "__main__":
	main()