import os
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import Counter


class Sentences_Dataset(torch.utils.data.Dataset):
    def __init__(self, raw_data, params:dict, n_grams:bool=False, is_training:bool=True):
        super(Sentences_Dataset, self).__init__()
        self.data = None                                                                    # Container for the data: [([tokens],[labels])]
        self.prepare_training_data(data=raw_data, training_data=is_training)

        if n_grams == True:
            self.n_gram_window_size = params["n_grams_size"]
            self.window_shifting = params["n_grams_shifting"]
        self.max_samp_length = max(len(tup[0]) for index, tup in enumerate(self.data))
        self.min_samp_length = min(len(tup[0]) for index, tup in enumerate(self.data))
        self.words_lengths_occurrency = self.word_occurency_counter()

        # Encoded and Decoded version of the data
        self.encoded_data = None
        self.decoded_data = None

    # Override of the Length Method
    def __len__(self):
        return len(self.data)

    # Override of the Retriever Item Method
    def __getitem__(self, idx):
        return self.data[idx]

    def prepare_training_data(self, data:dict, training_data:bool=True) -> list:
        self.data = []
        for sample in data:
            x = []
            y = []
            for (i,j) in zip(sample["tokens"], sample["labels"]):
                # In order to reduce the length of the longest strings, I decided to use the '.' character as delimiter.
                if training_data==True and i == ".":
                    x.append(i.lower())
                    y.append(j)
                    self.data.append((x,y))
                    x = []
                    y = []
                else:
                    x.append(i.lower())
                    y.append(j)
            if x != [] and y != []:
                self.data.append((x,y))                     # list of tuples of lists(i.e tokens and labels)

    # ----- Randomize <unk> N-Grams
    def randomize_n_grams(self, unk_token, n:int=50):
        unk_lines = np.random.permutation(len(self.data))[:n]
        with tqdm(range(len(self.data)), desc="<unk> random") as pbar:
            for i, row in zip(pbar, self.data):
                if (None not in row[0] or "<unk>" not in row[0]) and i in unk_lines:
                    rand_idx = random.choice([ tok for tok in range(len(row[0]))])
                    self.data[i][0][rand_idx] = unk_token
        print("-- '<unk>' Token Randomness Added --")
    def randomize_encoded_n_grams(self, unk_token, n:int=50):
        unk_lines = np.random.permutation(len(self.data))[:n]
        with tqdm(range(len(self.encoded_data)), desc="<unk> random") as pbar:
            for i, row in zip(pbar, self.encoded_data):
                if (1 not in row[0]) and i in unk_lines:
                    rand_idx = random.choice([ tok for tok in range(len(row[0]))])
                    self.encoded_data[i][0][rand_idx] = unk_token
        print("-- '<unk>' Token Randomness Added --")

    # ----- Create the N-Grams
    def n_grams_creation(self, params, save_result:bool=True, reduced:bool=False):
        discarded_list = []
        n_gram_data = []

        end_sentence_counter = 0
        samples_counter = 0
        discarded_counter = 0
        with tqdm(range(len(self.data)), desc=f"{self.n_gram_window_size}-Grams creation") as pbar:
            for i, row in zip(pbar, self.data):
                token_window = []
                label_window = []
                for i in range(0, len(row[0]), self.window_shifting):
                    token_window = row[0][i:i+self.n_gram_window_size]
                    label_window = row[1][i:i+self.n_gram_window_size]
                    if len(token_window) < self.n_gram_window_size:
                        while len(token_window) < self.n_gram_window_size:
                            token_window.append(None)
                            label_window.append(None)
                    if reduced == True:
                        if token_window[0] == "." and end_sentence_counter == 0:
                            if all(elem == None for elem in token_window[1:len(token_window)]):
                                print("Last 'End Sentences Chunk' Added Correctly")
                                n_gram_data.append((token_window,label_window))
                                samples_counter += 1
                                end_sentence_counter += 1
                                discarded_list.append((token_window,label_window))
                        elif all((rows[0], rows[1]) != (token_window, label_window) for rows in discarded_list):
                            if all((rows[0], rows[1]) != (token_window, label_window) for rows in n_gram_data[i:len(n_gram_data)]):
                                n_gram_data.append((token_window,label_window))
                                samples_counter += 1
                            else:
                                discarded_counter += 1
                                discarded_list.append((token_window,label_window))
                                continue
                        else:
                            discarded_counter += 1
                            continue
                    else:
                        n_gram_data.append((token_window,label_window))
                        samples_counter += 1
                pbar.set_postfix(SAMPLES=samples_counter,
                                 DISCARDED_SAMPLES = discarded_counter,
                                 DISCARDED = discarded_list[-1] if len(discarded_list)>0 else None,
                                 LEN_DISCARDED_LIST = len(discarded_list))
            pbar.update(0)
            if all((rows[0], rows[1]) for rows in n_gram_data):
                print("- There're some Identical Rows")
            else:
                print("- There're no Identical Rows")
        self.data = n_gram_data
        self.max_samp_length = max(len(tup[0]) for index, tup in enumerate(self.data))
        self.min_samp_length = min(len(tup[0]) for index, tup in enumerate(self.data))
        if save_result == True:
            # Save the N-Grams
            #  Avoid the repetion of the creation process that could be honerous in terms of time
            if reduced==True:
                name = f"{params['n_grams_size']}_gram_{params['n_grams_shifting']}.ngram"
            else:
                name = f"{params['n_grams_size']}_gram_{params['n_grams_shifting']}.engram"
            torch.save({"n-gram": n_gram_data}, os.path.join(params["n_grams_path"],name))
            print("- N-Grams Saved")

    def n_grams_loading(self, n_grams_path:str, file:str):
        load_n_gram = torch.load(os.path.join(n_grams_path,file))["n-gram"]
        print("-- N-Grams Loaded --")
        self.data = load_n_gram
        self.max_samp_length = max(len(tup[0]) for index, tup in enumerate(self.data))
        self.min_samp_length = min(len(tup[0]) for index, tup in enumerate(self.data))
        self.words_lengths_occurrency = self.word_occurency_counter()

    def encode_data(self, word2id, label2id, max_length):
        self.encoded_data = []
        with tqdm(range(len(self.data)), desc="Encode data") as pbar:
            for i, row in zip(pbar, self.data):
                x = []
                y = []
                for word, label in zip(row[0], row[1]):
                    if word not in word2id.keys() and word != None:
                        x.append(word2id["<unk>"])
                        y.append(label2id[label])
                    elif word == None:
                        x.append(word2id["<pad>"])
                        y.append(label2id["<pad>"])
                    else:
                        x.append(word2id[word])
                        y.append(label2id[label])
                if len(x) < max_length:
                    x, y = self.padding_data(x, y, max_length,
                                            pad_symbol=[word2id["<pad>"],label2id["<pad>"]])
                self.encoded_data.append((x,y))
        print("- Encoded Completed")
        self.encoded_data = torch.tensor(self.encoded_data)

    def decode_data(self, id2word, id2label):
        self.decoded_data = []
        for row in self.data:
            x = []
            y = []
            for word, label in zip(row[0], row[1]):
                if word not in id2word.keys():
                    x.append(id2word["<unk>"])
                    y.append(id2label[label])
                else:
                    x.append(id2word[word])
                    y.append(id2label[label])
            self.decoded_data.append((x,y))
        print("-- Decoded Completed --")
        self.decoded_data = torch.tensor(self.decoded_data)

    def padding_data(self, x, y, max_length, pad_symbol:list):
        while len(x) < max_length:
            x.append(pad_symbol[0])
            y.append(pad_symbol[1])
        return x, y

    def word_occurency_counter(self):
        lengths = []
        for index, tup in enumerate(self.data):
            lengths.append(len(tup[0]))
        words_lengths_occurrency = Counter(sorted(lengths))
        return words_lengths_occurrency

    def elements_to_be_removed(self, min_length=1, max_length=1):
        counter = 0
        self.elem_to_be_removed = []
        for sample in self.data:
            if len(sample[0]) >=min_length and len(sample[0]) <=max_length:
                # Check if all tokens are labelled as "O"
                true_or_false = all(elem == "O" for elem in sample[1])
                if true_or_false == True:
                    self.elem_to_be_removed.append(counter)
            counter += 1

    def remove_elements_from_list(self):
        # Update the Data List
        self.data = [tup for i, tup in enumerate(self.data) if i not in self.elem_to_be_removed]
        # Update the Other Atributes
        self.max_samp_length = max(len(tup[0]) for index, tup in enumerate(self.data))
        self.min_samp_length = min(len(tup[0]) for index, tup in enumerate(self.data))
        self.words_lengths_occurrency = self.word_occurency_counter()