from transformers import AlbertModel
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
##todo: INSTALL PYTORCH

class ModelBuilder:
    '''build, compile and train the model according to the given parameters'''
    def __init__(self,
                 dropout=0.5,
                 learningRate=12,
                 batchSize = 9,
                 epochs = 20
                ):
        ##todo: ADD VALUE AND NULL CHECKS
        #ratio of nodes turned off during training
        self.Dropout = dropout              
        #rate at which model weights are updated
        self.LearningRate = learningRate    
        #Nb of epochs for training
        self.Epochs = epochs
        #bacth size for training
        self.BatchSize = batchSize
        ##ALBERT is a pretrained model for NLP tasks
        self.AlbertModel = AlbertModel.from_pretrained('albert-xxlarge-v2')
        self.IsCompiled = False

    def CreateModel(self):
        #the output of Albert model is passed through a decision tree (dense layers) to determine hate speech. Dropout to avoid overfitting
        model = keras.Sequential()
        model.add(self.AlbertModel)
        model.add(layers.Dense(128,activation='relu'))
        model.add(layers.Dropout(self.Dropout))
        model.add(layers.Dense(64,activation='relu'))
        model.add(layers.Dropout(self.Dropout))
        model.add(layers.Dense(32,activation='relu'))
        model.add(layers.Dropout(self.Dropout))
        model.add(layers.Dense(2,activation='sigmoid'))
        self.Model = model

    def Compile(self):
        if self.Model is not None:
            self.Model.compile(Adam(lr=self.LearningRate), loss='binary_crossentropy', metrics=['accuracy'])
            self.IsCompiled = True
        
    def Train(self,trainX,trainY,validX,validY):
        if  self.IsCompiled:
            history = self.Model.fit(trainX, trainY, validation_data=(validX, validY), batch_size=self.BatchSize, epochs=self.Epochs)
            return history

        return None