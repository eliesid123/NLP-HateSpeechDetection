import numpy as np
from transformers import AlbertTokenizer

class Tokenizor():
    def __init__(self,maxLen=64) -> None:
        self.Tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
        self.MaxLen = maxLen

    def EncodeAll(self,data) :
        input_ids = []
        attention_masks = []

        for i in range(len(data)):
            encoded = self.EncodeSentece(data[i])
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        return np.array(input_ids),np.array(attention_masks)

    def EncodeSentece(self,line):
        encoded = self.Tokenizer.encode_plus(
              line,
              add_special_tokens=True,
              max_length=self.MaxLen,
              padding = "longest",
              return_attention_mask=True,
            )
        return encoded