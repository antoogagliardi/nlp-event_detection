import os
import re
import json
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
from pprint import pprint


# Read configuration file
def read_config_file(config_path:str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


## Folder Preparing
def prepare_base_folder():
    os.makedirs("ckpt", exist_ok=True)                  # Training Checkpoints of the model
    os.makedirs("utils", exist_ok=True)                 # Utilities: Hyper-Params, GloVe Files, ...
    os.makedirs("utils/dictionaries", exist_ok=True)    # Dictionarie of the model
    os.makedirs("utils/ngrams", exist_ok=True)          # N-Grams Division


## JSONL Data Reading Functions
def load_json_data(data_path) -> list:
    result = []
    for line in open(data_path, "r"):
        result.append(json.loads(line))
    return result

def print_BIO_example(data, n_examples):
    counter = 0
    for sample in data:
        if counter < n_examples:
            print(f"------ Sample {counter}")
            for (i,j) in zip(sample["tokens"],sample["labels"]):
                print(f"({i.lower()},{j})")
            print("---------------------------------")
        counter += 1


## DataLoader Utility
def look_into_DataLoader(dataset:DataLoader):
    # Look into the DataLoader
    print("N° of Batches:", len(dataset))
    train_features = next(iter(dataset))
    print(f"Single Batch Shape: {train_features.size()}")
    print(f"N° of Sample per Batch: {train_features.size()[0]}")

    sample = train_features.squeeze()
    print(f"Samples in the single Batch:\n {sample}")


## Inspect Training Checkpoints
def inspect_checkpoints(checkpoint_dir:str):
    max_checkpoint = 0
    x = int()
    for filename in os.listdir(checkpoint_dir):
        if ".pt" in filename:
            x = int(re.search("[0-9]+", filename).group())
        if x >= max_checkpoint:
            max_checkpoint = x
    return max_checkpoint


## Dataset Analysis
def label_distribution(data):
    # Put each Label_Token into a list
    labels = []
    for index, row in enumerate(data):
        for elem in row[1]:
            labels.append(elem)

    # Total number of Label_Token
    length_data = len(labels)

    # Count the frequency of each Label_Token
    labels_distribution = Counter(sorted(labels))

    distribution_scaled = {}
    with tqdm(range(len(labels_distribution.items())), desc="Retrieve distribution of labels") as pbar:
        for _, (key,val) in zip(pbar, labels_distribution.items()):
            distribution_scaled[key] = val/length_data

    return length_data, labels_distribution, distribution_scaled

def classes_weigths_creation(length_data:int, labels_idx:dict, class_percentage:dict):
    classes_weights = np.empty(shape=(0),dtype=np.float32)
    with tqdm(range(len(labels_idx)), desc="Class weights creation") as pbar:
        for _, (key,val) in zip(pbar, labels_idx.items()):
            epsilon = 1.0 / (length_data * np.sqrt(class_percentage[key]*length_data))
            # Calculate the inverse of the frequency of each class
            weight = int(1.0 / (class_percentage[key] + epsilon))
            classes_weights = np.append(classes_weights, [weight], axis=0)
    pprint([f"{val} -> weigth = {weight}" for (_, val), weight in zip(labels_idx.items(), classes_weights)])
    classes_weights = torch.from_numpy(classes_weights).type(torch.FloatTensor)
    return classes_weights


## GloVe Pretrained Word Embedding Functions
# ---------------------------------------------------------------------
def check_out_words(data, glove_words):
    unk_words = []
    with tqdm(range(len(data))) as pbar:
        for idx, sample in zip(pbar, data):
            for (i,j) in zip(sample[0], sample[1]):
                if None not in sample[0]:
                    if i.lower() not in glove_words:
                        unk_words.append(i)
    if unk_words != []:
        print("- Length of Out Words: ", len(unk_words))
        print("- Out Words: ", unk_words)
    else:
        print("-- No Unknown Words found -- ")
# ---------------------------------------------------------------------

# Legge l'intero GloVe file
def read_glove_embeddings(data_path) -> dict:
    result = {}
    with open(data_path, "r") as file:
        lines = file.readlines()
        with tqdm(range(len(lines)), desc="Reading GloVe file") as pbar:
            for i, line in zip(pbar, lines):
                line = line.replace("\n","")
                line = re.split(string=line, pattern=" ")
                result[line[0]] = [val for val in line[1:len(line)]]
    return result

def retrieve_glove_myDictionary(glove_embed_dict, glove_words, my_dict, max_length:int):
    result = {}
    counter_pad = 0
    counter_unk = 0
    with tqdm(range(len(my_dict)), desc="Retrieve GloVe Embed") as pbar:
        for _, (key,val) in zip(pbar, my_dict.items()):
            if val == "<pad>" and counter_pad < 1:
                result[key] = np.round(np.random.uniform(low=-1.0/max_length,
                                                        high=1.0/max_length,
                                                        size=max_length), 6).tolist()
                counter_pad += 1
                continue
            if val == "<unk>" and counter_unk < 1:
                result[key] = np.round(np.random.uniform(low=-1.0/max_length,
                                                        high=1.0/max_length,
                                                        size=max_length), 6).tolist()
                counter_unk += 1
                continue
            if (val != "<pad>" or val != "<unk>") and val in glove_words:
                result[key] = glove_embed_dict[val]
    return result


## Vocabulary Functions
# Note: word_to_idx() è concepita per escludere tutti quei token del dataset che sono
#       sconosciuti in GloVe. Questo porterà poi alla loro codifica in <unk> al momento del training
def word_to_idx(data, target_vocabulary:list=None):
    word_to_idx = {"<pad>": 0,
                   "<unk>": 1
                  }
    with tqdm(range(len(data)), desc="Word-To-Idx") as pbar:
        for _, (x,y) in zip(pbar, data):
            for word in x:
                # Se non stiamo usando un target vocabulary (e.g uno estratto da GloVe)
                if target_vocabulary == None:
                    # Se la parola non è già stata aggiunta al dizionario
                    if word not in word_to_idx.keys():
                        word_to_idx[word] = len(word_to_idx)
                else:
                    # Se la parola è nel target vocabulary di GloVe e non è già stata aggiunta al dizionario
                    if word in target_vocabulary and word not in word_to_idx.keys():
                        word_to_idx[word] = len(word_to_idx)
    return word_to_idx

def idx_to_word(vocabulary_idx):
    idx_to_word = {}
    with tqdm(range(len(vocabulary_idx)), desc="Idx-To-Word") as pbar:
        for _, (key,val) in zip(pbar, vocabulary_idx.items()):
            if val not in idx_to_word.keys():
                idx_to_word[val] = key
    return idx_to_word

def idx_to_label(vocabulary_labels):
    idx_to_labels = {}
    with tqdm(range(len(vocabulary_labels)), desc="Idx-To-Label") as pbar:
        for _, (key, val) in zip(pbar, vocabulary_labels.items()):
            if val not in idx_to_labels.keys():
                idx_to_labels[val] = key
    return idx_to_labels

##########################################################################################

def check_vocabulary(dictionary:dict, n_examples:int=1):
    counter = 0
    for i, entry in zip(range(len(dictionary)), dictionary.items()):
        if counter <= n_examples:
            print(f"Index: {i} -> {entry}")
            counter += 1