from statistics import mode
from albert.modeling import AlbertConfig, AlbertModel

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ModelBuilder:
    def __init__(self,
                 batch_size=13,
                 is_training=True,
                 embedding_size=32,
                 hidden_size=32,
                 num_hidden_layers=5,
                 num_attention_heads=4,
                 intermediate_size=37,
                ):
        self.BatchSize = batch_size
        self.IsTraining = is_training
        self.EmbeddingSize = embedding_size
        self.HiddenSize = hidden_size
        self.NbHiddenLayers = num_hidden_layers
        self.NbAttentionHeads = num_attention_heads
        self.IntermediateSize = intermediate_size

    def _ini(self):
        self.AlbertConfig = AlbertConfig(
            embedding_size=self.EmbeddingSize,
            hidden_size=self.HiddenSize,
            num_hidden_layers=self.NbHiddenLayers,
            num_attention_heads=self.NbAttentionHeads,
            intermediate_size=self.IntermediateSize,
        )        
        self.AlbertModel = AlbertModel(
          config=self.AlbertConfig,
          is_training=self.IsTraining,
        )
    def CreateModel(self):
        model = keras.Sequential()
        model.add(self.AlbertModel)
        model.add(layers.Dense())