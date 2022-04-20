import imp
import torch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers  import Embedding
from tensorflow.keras.preprocessing import sequence

class CnnModel:
    def __init__(self,
                 dataManager,
                 dropout=0.5,
                 learningRate=0.01,
                 batchSize = 4,
                 epochs = 5,
                 validationSplit = 0.2,
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
        ##ALBERT is a pretrained model for NLP tasks
        self.DataManager = dataManager
        self.IsCompiled = False
        self.Model = None
        self.ModelSummary = None

    def Create(self, PRINT = False):
        #the output of Albert model is passed through a decision tree (dense layers) to determine hate speech. Dropout to avoid overfitting
        
        model = Sequential()
        model.add(Embedding(self.DataManager.VocabSize+1, 32, input_length=self.DataManager.MaxLength))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))
        self.Model = model
        self.ModelSummary = model.summary()
        if PRINT:
            print(self.ModelSummary)

    def Compile(self):
        if self.Model is not None:
            self.Model.compile(Adam(lr=self.LearningRate), loss='binary_crossentropy', metrics=['accuracy'])
            self.IsCompiled = True

    def Train(self,trainX,trainY,testX,testY):
        trainX = sequence.pad_sequences(trainX, maxlen=self.DataManager.MaxLength,dtype="int64")
        testX = sequence.pad_sequences(testX, maxlen=self.DataManager.MaxLength,dtype="int64")
        history = self.Model.fit(trainX, trainY, epochs=self.Epochs, batch_size=self.BatchSize)
        # Final evaluation of the model
        scores = self.Model.evaluate(testX, testY, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        return history

