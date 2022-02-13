import csv
import numpy as np 
from re import L, S
from nltk.corpus import stopwords 
import emoji
from nltk.util import ngrams, everygrams

class TrainingData():
	"""data and tags used for trainig"""
	def __init__(self, data, tags):
		super(TrainingData, self).__init__()
		self.Data = data
		self.Tags = tags
	
	def GetPair(self, i):
		return self.Data[1], self.Tags[i]

class DataSetManager():
	"""Reads and prepare raw data from csv files for training"""
	def __init__(self, path,n):
		super(DataSetManager, self).__init__()
		self.CsvPath = path
		self.NGramMax = n
		self._init()

	def _init(self):
		self.GetRawData()
		self._validSymbols = dict.fromkeys(map(ord, "!?\.,"), None)
		self._unvalidSymbols = dict.fromkeys(map(ord, "@#$%^&*<>/][}{"), None)
		self._stopWords = list(stopwords.words('english'))

	def GetRawData(self):
		with open(self.CsvPath,newline='',encoding='utf-8') as csvFile:
			rawData =  csv.reader(csvFile,delimiter='\t')
			next(rawData)
			data = list()
			tags = list()
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
			line = line.translate(self._validSymbols)
			words = line.split(' ')
			toRemove = list()
			for word in words:
				newWord = word.translate(self._unvalidSymbols)
				newWord = emoji.demojize(newWord) 
				if newWord != word or word == "URL" or len(word)<=1 or self._stopWords.count(word) >0:
					toRemove.append(word)
			self.TrainingData.Data[iter] = ' '.join([valid for valid in words if valid not in toRemove ])
			iter = iter+1

	def GetEveryGram(self):
		self.EveryGrams = list()
		for i in range(len(self.TrainingData.Data)):
			line, tag = self.TrainingData.GetPair(i)
			everyGrams = list(everygrams(line,self.NGramMax))
			self.EveryGrams.append(TrainingData(everyGrams),tag*np.ones(len(everyGrams)))
				 