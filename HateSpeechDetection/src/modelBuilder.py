import json
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import concatenate
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
                 tokenizer = None,
                 maxLength = 580,
                 vocabSize = 50000,
                 dropout=0.5,
                 learningRate=0.01,
                 batchSize = 4,
                 epochs = 5,
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
        self.MaxLength = maxLength
        self.VocabSize = vocabSize
        self.IsCompiled = False
        self.Model = None
        self.ModelSummary = None

        ################# DO NOT TOUCH #################
        self._modelJSONFile = ""
        self._modelWeightsFile = ""
        self._modelTokenizor = ""
        ################################################

    def Create(self, PRINT = False):
        # firstChannel = Sequential()
        # firstChannel.add(Embedding(self.VocabSize, 64, input_length=self.MaxLength))
        # firstChannel.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        # firstChannel.add(Dropout(0.3))
        # firstChannel.add(MaxPooling1D(pool_size=2))
        # firstChannel.add(LSTM(128))
        # firstChannel.add(Flatten())
        # secondChannel = Sequential()
        # secondChannel.add(Embedding(self.VocabSize, 64, input_length=self.MaxLength))
        # secondChannel.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        # secondChannel.add(Dropout(0.3))
        # secondChannel.add(MaxPooling1D(pool_size=2))
        # secondChannel.add(LSTM(128))
        # secondChannel.add(Flatten())
        # merged = concatenate([firstChannel,secondChannel])
        # model.add(merged)
        self.Tokenizer = Tokenizer()
        model = Sequential()
        model.add(Embedding(self.VocabSize, 64, input_length=self.MaxLength))
        model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(128))
        model.add(Flatten())
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1,activation='sigmoid'))
        self.Model = model
        self.ModelSummary = model.summary()
        if PRINT:
            print(self.ModelSummary)

    def encode_text(self, lines, length):
        encoded = self.Tokenizer.texts_to_sequences(lines)
        padded = pad_sequences(encoded, maxlen=length, padding='post')
        return padded
    def FitTokenizor(self,lines):
        self.Tokenizer.fit_on_texts(lines)
        self.MaxLength = max([len(s.split()) for s in lines])
        self.VocabSize = len(self.Tokenizer.word_index) + 1

    def Compile(self):
        if self.Model is not None:
            self.Model.compile(Adam(lr=self.LearningRate), loss='binary_crossentropy', metrics=['accuracy'])
            self.IsCompiled = True

    def Train(self,trainX,trainY,testX,testY):
        trainX_encoded = self.Tokenizer.Encode(trainX)
        testX_encoded = self.Tokenizer.Encode(testX)
        trainX_encoded = pad_sequences(trainX_encoded, maxlen=self.MaxLength,dtype="int64")
        testX_encoded = pad_sequences(testX_encoded, maxlen=self.MaxLength,dtype="int64")
        history = self.Model.fit(trainX_encoded, trainY, epochs=self.Epochs, batch_size=self.BatchSize)
        self.SaveModel()
        scores = self.Model.evaluate(testX_encoded, testY, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        return history

    def SaveModel(self):
        model_json = self.Model.to_json()
        with open(self._modelJSONFile, "w") as json_file:
            json_file.write(model_json)
        with open(self._modelTokenizor,"r") as tokenizorFile:
            json.dump(tokenizorFile,indent=2)
        self.Model.save_weights(self._modelWeightsFile)

    def LoadModel(self):
        with open(self._modelJSONFile, 'r') as json_file:
            loaded_model_json = json_file.read()
        self.Model = model_from_json(loaded_model_json)
        self.Model.load_weights(self._modelWeightsFile)
        with open(self._modelTokenizor,'r') as tokensFile:
            self.Tokenizer = tokensFile.load()
