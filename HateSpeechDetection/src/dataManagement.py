import csv
import json
from re import S
from nltk.corpus import stopwords 
import emoji
import numpy as np
from transformers import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

class TrainingData():
	"""data and tags used for trainig"""
	def __init__(self, data, tags):
		super(TrainingData, self).__init__()
		self.Data = data
		self.Labels = tags
		self.LabeledData = None
		self._ini()

	def _ini(self):
		self.Size = len(self.Data)
		self.MaxLength = 0
		self.LabeledData = dict(zip(self.Data,self.Labels))
		for line in self.Data:
			self.MaxLength = max(self.MaxLength,len(line))
	
	def GetPair(self, i):
		return self.Data[i], self.Labels[i]

	def Update(self):
		self.LabeledData = dict(zip(self.Data,self.Labels))
	
class DataSetManager():
	"""Reads and prepare raw data from csv files for training"""
	def __init__(self, path,cleanDataPath, tokenizor = None,dataSplit=0.8,csvFile = None):
		super(DataSetManager, self).__init__()
		self.CsvPath = path
		self.CleanDataPath = cleanDataPath
		self.TestSplit = dataSplit
		self.CsvFile = csvFile
		self.Tokenizor = tokenizor
		self.TrainingData = None
		self._init()

	def _init(self):
		self.GetRawData()
		self.MaxLength = self.TrainingData.MaxLength
		self.DatasetSize = self.TrainingData.Size
		#training data will have no symbols at all. words containing unvalid symbols are removed, and valid symbols are removed while keeping the word
		self._validSymbols = dict.fromkeys(map(ord, "!?\.,"), None)
		self._unvalidSymbols = dict.fromkeys(map(ord, "@#$%^&*<>/][}{"), None)
		#stopwords are words irrelevant to the meaning of the sentence (I,for,where...) , and needs to be removed to have better results
		self._stopWords = list(stopwords.words('english'))

	def GetRawData(self):
		if self.CsvFile == None:
			self.CsvFile = open(self.CsvPath,newline='',encoding='utf-8')
		with self.CsvFile as csvFile:
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

	def CleanData(self,isSave = False):
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
				if newWord != word or word == "URL" or len(word)<=1:# or self._stopWords.count(word) >0:
					toRemove.append(word)
			#reconstruct the sentence using only the useful words
			self.TrainingData.Data[iter] = ' '.join([valid for valid in words if valid not in toRemove ])
			iter = iter+1
		if isSave:
			self.SaveCleanData()

	def SaveCleanData(self):
		self.TrainingData.Update()
		jsonData = json.dumps(self.TrainingData.LabeledData,indent=2)
		with open(self.CleanDataPath, "w") as outfile:
			outfile.write(jsonData)

	def LoadCleanData(self):
		with open(self.CleanDataPath, 'r') as openfile:
			self.TrainingData.LabeledData = json.load(openfile)

	#take a portion of data for training and leave the rest for testing
	def GetTrainingData(self):
		return self.TrainingData.Data[0:(int)(self.TestSplit*self.TrainingData.Size)], np.array(self.TrainingData.Labels[0:(int)(self.TestSplit*self.TrainingData.Size)])	

	def GetValidationData(self):
		return self.TrainingData.Data[(int)(self.TestSplit*self.TrainingData.Size):], np.array(self.TrainingData.Labels[(int)(self.TestSplit*self.TrainingData.Size):])	

	def GetAllData(self):
		return self.TrainingData.Data, np.array(self.TrainingData.Labels)	
