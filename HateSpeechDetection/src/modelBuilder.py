from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers  import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer  
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CnnModel:
    def __init__(self,
                 maxLength = 580,
                 vocabSize = 50000,
                 dropout=0.5,
                 learningRate=0.01,
                 batchSize = 12,
                 epochs = 20,
                 validationSplit = 0.2
                ):
        #ratio of nodes turned off during training
        if dropout >1:
            print("Dropout can' be greater than 1, set to 0.5")
            self.Dropout = 0.5              
        else:
            self.Dropout = dropout              
        #rate at which model weights are updated
        if learningRate >0.5:
            print("Learning rate can' be greater than 0.5, set to 0.02")
            self.LearningRate = 0.02              
        else:
            self.LearningRate = learningRate              
        #Nb of epochs for training
        if epochs <0 :
            print("Epochs should be greater than 0, set to 10")
            self.Epochs = 10
        else:
            self.Epochs = epochs
        #split ratio between training and validation data
        if validationSplit >1:
            print("validationSplit can' be greater than 1, set to 0.2")
            self.ValidationSplit = 0.2              
        else:
            self.ValidationSplit = validationSplit    
        #bacth size for training
        if batchSize <0 :
            print("batchSize should be greater than 0, set to 8")
            self.BatchSize = 8
        else:
            self.BatchSize = epochs
        self.Tokenizer = Tokenizer()
        self.MaxLength = maxLength
        self.VocabSize = vocabSize
        self.IsCompiled = False
        self.Model = None
        self.ModelSummary = None

        ################# DO NOT TOUCH #################
        self._modelDir = "/home/elie/Desktop/NLP/HateSpeechDetection/models/other/"
        self._modelJSONFile = self._modelDir +  "modelJSON.json"
        self._modelWeightsFile = self._modelDir + "modelWEIGHTs.h5"
        self._modelTokenizor = ""
        ################################################

    def Create(self, PRINT = False):
        model = Sequential()
        model.add(Embedding(self.VocabSize, 16, input_length=self.MaxLength))
        model.add(Conv1D(filters=16, kernel_size=2, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1,activation='sigmoid'))

        self.Model = model
        self.ModelSummary = model.summary()
        if PRINT:
            print(self.ModelSummary)

    def EncodeText(self, lines):
        encoded = self.Tokenizer.texts_to_sequences(lines)
        padded = pad_sequences(encoded, maxlen=self.MaxLength, padding='post')
        return padded
        
    def FitTokenizor(self,lines):
        self.Tokenizer.fit_on_texts(lines)
        self.MaxLength = max([len(s.split()) for s in lines])
        self.VocabSize = len(self.Tokenizer.word_index) + 1

    def Compile(self,data=None):
        if self.Model is not None:
            self.Model.compile(Adam(lr=self.LearningRate), loss='binary_crossentropy', metrics=['accuracy'])
            self.IsCompiled = True
        if data is not None:
            self.FitTokenizor(data)

    def Train(self,trainX,trainY,testX,testY):
        trainX_encoded = self.EncodeText(trainX)
        testX_encoded = self.EncodeText(testX)
        # trainX_encoded = pad_sequences(trainX_encoded, maxlen=self.MaxLength,dtype="int64")
        # testX_encoded = pad_sequences(testX_encoded, maxlen=self.MaxLength,dtype="int64")
        history = self.Model.fit(trainX_encoded, trainY, epochs=self.Epochs, batch_size=self.BatchSize)
        self.SaveModel()
        scores = self.Model.evaluate(testX_encoded, testY, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        return history

    def SaveModel(self):
        model_json = self.Model.to_json()
        with open(self._modelJSONFile, "w") as json_file:
            json_file.write(model_json)
        # with open(self._modelTokenizor,"w") as tokenizorFile:
        #     json.dump(tokenizorFile,indent=2)
        self.Model.save_weights(self._modelWeightsFile)

    def LoadModel(self):
        with open(self._modelJSONFile, 'r') as json_file:
            loaded_model_json = json_file.read()
        self.Model = model_from_json(loaded_model_json)
        self.Model.load_weights(self._modelWeightsFile)
        # with open(self._modelTokenizor,'r') as tokensFile:
        #     self.Tokenizer = tokensFile.load()
