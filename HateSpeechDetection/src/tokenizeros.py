import numpy as np
from transformers import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

class Tokenizor():
    def __init__(self) -> None:
        self.Dictionary = dict()
        self.Index = 0

    def Encode(self,data) :
        encodedData = []
        for i in range(len(data)):
            encoded = self.EncodeSentece(data[i])
            encodedData.append(encoded)
        return np.array(encodedData)

    def EncodeSentece(self,line):
        words = line.split(" ")
        tokens = list()
        for word in words:
            tokens.append(self.EncodeWord(word.lower()))    
        return tokens

    def EncodeWord(self,word):
        if word in self.Dictionary.values():
            return self._getKey(word)
        else:
            self.Index +=1
            self.Dictionary.update({self.Index:word})
            return self.Index

    def _getKey(self,val):
        for key, value in self.Dictionary.items():
            if val == value:
                return key