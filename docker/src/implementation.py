## ---- Import the class containing the ED Model definition from ED_Model.py ----
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import List

from model import Model

from .ed_model import ED_Model

"""
class ED_Model(nn.Module):
    def __init__(self, dictionaries:dict, hyperparams:dict):
        super(ED_Model,self).__init__()
        
        self.VOCABULARIES = dictionaries
        
        self.VOCABULARY_SIZE = len(dictionaries["word2idx"])
        self.LABELS_SIZE = len(dictionaries["label2idx"])
        self.EMBEDDING_LATENT_DIM = hyperparams["embed_latent_dimention"]
        self.LSTM_LATENT_DIM = hyperparams["lstm_latent_dimention"]
        
         # WORD EMBEDDING MATRIX (LOOK-UP TABLE)
        self.word_embedding = nn.Embedding(num_embeddings=self.VOCABULARY_SIZE,
                                           embedding_dim=self.EMBEDDING_LATENT_DIM).to(device=hyperparams["device"])
        # LSTM
        self.lstm = torch.nn.LSTM(input_size=self.EMBEDDING_LATENT_DIM, hidden_size=self.LSTM_LATENT_DIM,
                                  num_layers=2, bias=True, batch_first=True,
                                  dropout=0, bidirectional= False if hyperparams["bi_lstm"] is False else True).to(device=hyperparams["device"])
        # CLASSIFIER
        self.classifier = nn.Linear(in_features= self.LSTM_LATENT_DIM if hyperparams["bi_lstm"] is False else self.LSTM_LATENT_DIM*2,
                                    out_features=self.LABELS_SIZE).to(device=hyperparams["device"])
    
    def forward(self, input_sequence):
        # Embedding of the SENTENCES ONLY
        embeddings = self.word_embedding(input_sequence)
        
        output_features, _ = self.lstm(embeddings)
        
        #-- Dense Layer
        predicted_labels = self.classifier(output_features)
        
        #-- Activation Function
        labels_score = F.softmax(predicted_labels, dim=-1)
        
        return labels_score
"""


def build_model(device: str) -> Model:
    # return RandomBaseline()
    return TestModel()

class RandomBaseline(Model):
    options = [
        (22458, "B-ACTION"),
        (13256, "B-CHANGE"),
        (2711, "B-POSSESSION"),
        (6405, "B-SCENARIO"),
        (3024, "B-SENTIMENT"),
        (457, "I-ACTION"),
        (583, "I-CHANGE"),
        (30, "I-POSSESSION"),
        (505, "I-SCENARIO"),
        (24, "I-SENTIMENT"),
        (463402, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class TestModel(Model):
    def __init__(self, checkpoint_path="model/model.pt",
                 hyperparams_path="model/hyper_params.params",
                 vocabularies_path="model/vocabulary.vocab"):
        # LOADING HYPER-PARAMETERS
        print(checkpoint_path)
        print(hyperparams_path)
        print(vocabularies_path)
        self.hyperparams = torch.load(hyperparams_path, weights_only=True)["hyperparams"]
        dictionaries = torch.load(vocabularies_path, weights_only=True)["vocabularies"]

        # DEVICE CHECKING
        if torch.cuda.is_available() == True:   self.hyperparams["device"] = "cuda"
        else:                                   self.hyperparams["device"] = "cpu" 
        print("Hyper-Params: ", self.hyperparams)
        
        # MODEL INSTATIATION
        self.ed_model = ED_Model(dictionaries=dictionaries, hyperparams=self.hyperparams)

        # CHECKPOINT LOADING ----------------------------------------------------------------
        print("-- Loading Model from Checkpoint --")
        checkpoint_dict = torch.load(checkpoint_path, map_location=self.hyperparams["device"], weights_only=False)
        self.ed_model.load_state_dict(checkpoint_dict["model"])             #   Model's Weights
        print(" - Last Training Checkpoint Loaded")
        self.ed_model.VOCABULARIES = checkpoint_dict["dictionaries"]        #   Model's Dictionaries Loaded
        print(" - Model's Dictionaries Loaded")
        
        # Model in EVALUATION MODE
        self.ed_model.eval()
        print(" - Model in EVALUATION Mode")
        
        print("-- Model Loaded Correctly --")
        
    def encode_data(self, tokens):
        max_length = max(len(row) for row in tokens)
        encoded_data = []
        for row in tokens:
            x = []
            for word in row:
                if word not in self.ed_model.VOCABULARIES["word2idx"].keys() and word != None:
                    x.append(self.ed_model.VOCABULARIES["word2idx"]["<unk>"])
                elif word == None:
                    x.append(self.ed_model.VOCABULARIES["word2idx"]["<pad>"])
                else:
                    x.append(self.ed_model.VOCABULARIES["word2idx"][word])
            if len(x) < max_length:
                x = self.padding_data(x, max_length,
                                      pad_symbol=self.ed_model.VOCABULARIES["word2idx"]["<pad>"])
            encoded_data.append([x])
        encoded_data = torch.tensor(encoded_data)
        return encoded_data

    def decode_data(self, labels, real_sentence_length:int):
        decoded_data = []
        for row in labels:
            y = []
            for label in row:
                if len(y) < real_sentence_length:
                    y.append(self.ed_model.VOCABULARIES["idx2label"][label.cpu().item()])
            decoded_data.append(y)
        return decoded_data
    
    def padding_data(self, x, max_length, pad_symbol):
        while len(x) < max_length:
            x.append(pad_symbol)
        return x

    def predict(self, tokens:List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        #pass

        total_predictions : List[List[int]] = []
        
        # Deactivate Autograd
        sentences = self.encode_data(tokens)
        with torch.no_grad():
            for x, target in zip(sentences, tokens):
                # Load Sentences on the device
                x = x.to(device=self.hyperparams["device"])

                # Calling the Model to do its predictions
                model_output = self.ed_model(x)
                _ , top_label_indices = torch.max(model_output, -1)

                # Decode predictions made by the model
                top_label_indices = self.decode_data(top_label_indices,
                                                     real_sentence_length=len(target))
                for elem in top_label_indices:
                    total_predictions.append(elem)

        return total_predictions
