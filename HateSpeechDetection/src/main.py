from asyncore import read
import re
from dataReader import DataSetManager
import os

dir = os.getcwd()

def main():
	path = os.path.join(dir,"datasets","training-v1","offenseval-training-v1.tsv")
	reader = DataSetManager(path,5)
	# reader.CleanData()
	reader.GetEveryGram()

if __name__ == "__main__":
	main()