import torch
from transformers import AlbertConfig, AlbertModel
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

class ModelBuilder:
    '''build, compile and train the model according to the given parameters'''
    def __init__(self,
                 tokenizer,
                 dropout=0.5,
                 learningRate=12,
                 batchSize = 9,
                 epochs = 20,
                 validationSplit = 0.2,
                 inputSize = 64
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
        self.InputSize = inputSize
        ##ALBERT is a pretrained model for NLP tasks
        self.ValidationSplit = validationSplit
        self.AlbertModel = AlbertModel.from_pretrained('albert-xxlarge-v2')
        self.AlbertConfig = AlbertConfig()
        self.Tokenizer = tokenizer
        self.IsCompiled = False
        self.Model = None
        self.ModelSummary = None

    def Create(self, PRINT = False):
        #the output of Albert model is passed through a decision tree (dense layers) to determine hate speech. Dropout to avoid overfitting

        #INITIAL KAGGLE GUIDE
        # input_ids = tf.keras.Input(shape=(self.InputSize,),dtype='int64')
        # attention_masks = tf.keras.Input(shape=(self.InputSize,),dtype='float')
        # output, layer = self.AlbertModel(input_ids,attention_masks)

        #THIS IS BASED ON HUGGINGFACE DOCUMENTATION
        # tensor = np.ndarray(shape=(self.InputSize,),dtype='long')
        # input_ids = torch.tensor(tensor)
        # tensor = np.ndarray(shape=(self.InputSize,),dtype='float')
        # attention_masks = torch.tensor(tensor)
        # output = self.AlbertModel(input_ids,attention_masks)
        
        #THIS IS FROM HUGGINGFACE DOCUMENTATION
        # sampleText = "some line............................................."
        # encoded = self.Tokenizer.EncodeSentece(sampleText)
        # input_ids = encoded['input_ids']
        # attention_masks = encoded['attention_mask']
        # output = self.AlbertModel(torch.tensor(encoded).unsqueeze(self.BatchSize))

        output = self.AlbertModel(self.AlbertConfig)

        output = output[1]
        output = tf.keras.layers.Dense(128,activation='relu')(output)
        output = tf.keras.layers.Dropout(self.Dropout)(output)
        output = tf.keras.layers.Dense(64,activation='relu')(output)
        output = tf.keras.layers.Dropout(self.Dropout)(output)
        output = tf.keras.layers.Dense(32,activation='relu')(output)
        output = tf.keras.layers.Dropout(self.Dropout)(output)
        output = tf.keras.layers.Dense(1,activation='sigmoid')(output)
  
        model = tf.keras.models.Model(inputs = [input_ids, attention_masks] ,outputs = output)
 
        self.Model = model
        self.ModelSummary = model.summary()
        if PRINT:
            print(self.ModelSummary)

    def Compile(self):
        if self.Model is not None:
            self.Model.compile(Adam(lr=self.LearningRate), loss='binary_crossentropy', metrics=['accuracy'])
            self.IsCompiled = True
        
    def Train(self,trainX,trainY):
        if  self.IsCompiled:
            trainInput, trainMask = self.Tokenizer.Encode(trainX)
            history = self.Model.fit(
                                    x = [trainInput, trainMask],
                                    y = trainY,
                                    validation_split=self.ValidationSplit,
                                    batch_size=self.BatchSize, 
                                    epochs=self.Epochs)
            return history

        return None