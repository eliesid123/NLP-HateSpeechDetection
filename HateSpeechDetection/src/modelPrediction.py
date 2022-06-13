from tensorflow.keras.optimizers import Adam
import json
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer  
from tensorflow.keras.preprocessing.sequence import pad_sequences
class PredictionModel:
    def __init__(self,modelPath,paramsPath,cleanDataPath) -> None:
        self.ModelPath = modelPath
        self.ModelWeightsPath = paramsPath
        self.CleanDataPath = cleanDataPath
        self._ini()

    def _ini(self):
        with open(self.ModelPath, 'r') as json_file:
            loaded_model_json = json_file.read()
        self.Model = model_from_json(loaded_model_json)
        self.Model.load_weights(self.ModelWeightsPath)
        self.Tokenizer = Tokenizer()
        self.Compile()
        with open(self.CleanDataPath, 'r') as json_file:
            labeledData = json.load(json_file)
        self.FitTokenizor(labeledData.keys())

    def EncodeText(self, lines):
        encoded = self.Tokenizer.texts_to_sequences(lines)
        padded = pad_sequences(encoded, maxlen=self.MaxLength, padding='post')
        return padded

    def FitTokenizor(self,lines):
        self.Tokenizer.fit_on_texts(lines)
        self.MaxLength = max([len(s.split()) for s in lines])
        self.VocabSize = len(self.Tokenizer.word_index) + 1
    
    def Compile(self):
        if self.Model is not None:
            self.Model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            self.IsCompiled = True

    def Predict(self,line):
        encodedLine = self.EncodeText(line)
        prediction =  self.Model.predict(encodedLine)
        return prediction

    def Test(self,trainX,trainY):
        self.FitTokenizor(trainX)
        trainX_encoded = self.EncodeText(trainX)
        scores = self.Model.evaluate(trainX_encoded, trainY, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))