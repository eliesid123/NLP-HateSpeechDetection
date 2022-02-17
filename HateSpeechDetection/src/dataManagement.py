import csv
import numpy as np 
from re import L, S
from nltk.corpus import stopwords 
import emoji
import tensorflow as tf
from sklearn.model_selection import train_test_split

class TrainingData():
	"""data and tags used for trainig"""
	def __init__(self, data, tags):
		super(TrainingData, self).__init__()
		self.Data = data
		self.Labels = tags
		self.Size = len(data)
	
	def GetPair(self, i):
		return self.Data[1], self.Labels[i]

class DataSetManager():
	"""Reads and prepare raw data from csv files for training"""
	def __init__(self, path,n):
		super(DataSetManager, self).__init__()
		self.CsvPath = path
		self.TestSplit = n
		self._init()

	def _init(self):
		self.GetRawData()

		#training data will have no symbols at all. words containing unvalid symbols are removed, and valid symbols are removed while keeping the word
		self._validSymbols = dict.fromkeys(map(ord, "!?\.,"), None)
		self._unvalidSymbols = dict.fromkeys(map(ord, "@#$%^&*<>/][}{"), None)
		#stopwords are words irrelevant to the meaning of the sentence (I,for,where...) , and needs to be removed to have better results
		self._stopWords = list(stopwords.words('english'))

	def GetRawData(self):
		with open(self.CsvPath,newline='',encoding='utf-8') as csvFile:
			rawData =  csv.reader(csvFile,delimiter='\t')
			next(rawData)
			data = list()
			tags = list()
			#create binary tags (0 1) based on the csv dataset
			for row in rawData:
				data.append(row[1])
				if row[2]=='OFF':
					tags.append(1)
				else:
					tags.append(0)

		self.TrainingData = TrainingData(data,tags)
		
	def CleanData(self):
		iter = 0
		for line in self.TrainingData.Data:
			#remove valid symbols from sentences
			line = line.translate(self._validSymbols)
			words = line.split(' ')
			toRemove = list()
			for word in words:
				#remove unvalid symbols and emojis from the word then check if the word changes
				newWord = word.translate(self._unvalidSymbols)
				newWord = emoji.demojize(newWord) 
				#remove word if it is changed, 1 character, or is a stopword
				if newWord != word or word == "URL" or len(word)<=1 or self._stopWords.count(word) >0:
					toRemove.append(word)
			#reconstruct the sentence using only the useful words
			self.TrainingData.Data[iter] = ' '.join([valid for valid in words if valid not in toRemove ])
			iter = iter+1

	#take a portion of data for training and leave the rest for testing
	def GetTrainingData(self):
		textTrain = self.TrainingData.Data[0:(int)(self.TestSplit*self.TrainingData.Size)]
		labelTrain = self.TrainingData.Labels[0:(int)(self.TestSplit*self.TrainingData.Size)]	
		return train_test_split(textTrain,labelTrain,test_size=0.23)

	def GetTestingData(self):
		return self.TrainingData.Data[(int)(self.TestSplit*self.TrainingData.Size):], self.TrainingData.Labels[(int)(self.TestSplit*self.TrainingData.Size):]	