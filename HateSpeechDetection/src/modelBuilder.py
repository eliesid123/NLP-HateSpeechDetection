from transformers import AlbertConfig, AlbertModel, AlbertTokenizer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ModelBuilder:
    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=12,
                 intermediate_size=3072
                ):
        self.HiddenSize = hidden_size
        self.NbAttentionHeads = num_attention_heads
        self.IntermediateSize = intermediate_size

    def _ini(self):
        self.AlbertConfig = AlbertConfig(
            hidden_size=self.HiddenSize,
            num_attention_heads=self.NbAttentionHeads,
            intermediate_size=self.IntermediateSize,
        )        
        self.AlbertModel = AlbertModel.from_pretrained('albert-xxlarge-v2')

    def CreateModel(self):
        model = keras.Sequential()
        model.add(self.AlbertModel)
        model.add(layers.Dense())