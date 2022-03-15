import torch
from transformers import AlbertModel
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class ModelBuilder:
    '''build, compile and train the model according to the given parameters'''
    def __init__(self,
                 tokenizer,
                 dropout=0.5,
                 learningRate=0.01,
                 batchSize = 8,
                 epochs = 5,
                 validationSplit = 0.2,
                ):
        if tokenizer == None:
            print("Tokrnizer is None")
            return
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
        self.Tokenizer = tokenizer
        self.InputSize = tokenizer.MaxLen    
        ##ALBERT is a pretrained model for NLP tasks
        self.AlbertModel = AlbertModel.from_pretrained('albert-xxlarge-v2')
        self.IsCompiled = False
        self.Model = None
        self.ModelSummary = None

    def Create(self, PRINT = False):
        #the output of Albert model is passed through a decision tree (dense layers) to determine hate speech. Dropout to avoid overfitting
        
        output = self.GetAlbertOutput("sample text")
        inputLayer = tf.keras.Input(shape = output.shape)
        output = tf.keras.layers.Dense(128,activation='relu')(inputLayer)
        output = tf.keras.layers.Dropout(self.Dropout)(output)
        output = tf.keras.layers.Dense(64,activation='relu')(output)
        output = tf.keras.layers.Dropout(self.Dropout)(output)
        output = tf.keras.layers.Dense(32,activation='relu')(output)
        output = tf.keras.layers.Dropout(self.Dropout)(output)
        output = tf.keras.layers.Dense(1,activation='sigmoid')(output)
  
        model = tf.keras.models.Model(inputs = inputLayer ,outputs = output)
 
        self.Model = model
        self.ModelSummary = model.summary()
        if PRINT:
            print(self.ModelSummary)

    def Compile(self):
        if self.Model is not None:
            self.Model.compile(Adam(lr=self.LearningRate), loss='binary_crossentropy', metrics=['accuracy'])
            self.IsCompiled = True

    def GetAlbertOutput(self,line):
        encoded = self.Tokenizer.EncodeSentece(line)
        input_ids = torch.tensor(encoded['input_ids']).unsqueeze(0)
        attention_mask = torch.tensor(encoded['attention_mask']).unsqueeze(0)
        output = self.AlbertModel(input_ids,attention_mask)
        return output[0]

    def Train(self,trainX,trainY):
        if  self.IsCompiled:
            output = []
            for i in range(200):
                output.append(self.GetAlbertOutput(trainX[i]))
            history = self.Model.fit(
                                    x = output,
                                    y = trainY,
                                    validation_split=self.ValidationSplit,
                                    batch_size=self.BatchSize, 
                                    epochs=self.Epochs)
            return history

        return None