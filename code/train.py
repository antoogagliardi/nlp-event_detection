import os
import re
import random
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader


from src.utils import prepare_base_folder, read_config_file, load_json_data, read_glove_embeddings, retrieve_glove_myDictionary
from src.utils import word_to_idx, idx_to_word, idx_to_label, check_vocabulary
from src.utils import label_distribution, classes_weigths_creation
from src.data import Sentences_Dataset
from src.model import LSTM_Trainer



## Main Program
prepare_base_folder()

# Read configuration file
cfg = read_config_file("configs/config.yaml")
print("== Configuration File ==")
pprint(cfg)

# Setup the device for training
device = torch.device(cfg["project"]["device"])
print(f"- Device used: {device}")

# Prepare Folders and Paths
root_path = "../"
cwd = os.getcwd()
data_path = os.path.join(root_path, cfg["paths"]["data"])
    # Dataset Paths
train_set_path = os.path.join(data_path, "train.jsonl")
valid_set_path = os.path.join(data_path, "dev.jsonl")
test_set_path = os.path.join(data_path, "test.jsonl")
    # GloVe Pre-Trained
    ############################################################
    #  glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt #
    #  glove.42B.300d.txt                                      #
    #  glove.840B.300d.txt                                     #
    ############################################################
glove_folder = os.path.join(cwd, cfg["paths"]["glove_path"])
glove_size = cfg["project"]["glove"]["glove_size"]                  #   6,  42, 840
glove_embed_dim = cfg["project"]["glove"]["glove_embed_dim"]        # 100, 200, 300
glove_file = f"glove.{glove_size}B.{glove_embed_dim}d"
glove_pretrained_txt = os.path.join(glove_folder, (glove_file + ".txt"))
    # Utilities/Hyper-Parameters
utilities_path = os.path.join(cwd, cfg["paths"]["utils_path"])
    # N-Grams
ngrams_path = os.path.join(cwd, cfg["paths"]["ngrams_path"])
    # Vocabularies
vocabulary_path = os.path.join(cwd, cfg["paths"]["dict_path"])
    # Checkpoints
checkpoint_path = os.path.join(cwd, cfg["training"]["ckpt_path"])



# Hyper-Parameters of the experiment
RECREATE_HYPERPARAMS = cfg["project"]["hp"]["recreate"]
HYPERPARMS_ID = cfg["project"]["hp"]["id"]
if RECREATE_HYPERPARAMS == True:
    hyperparams = {"n_grams_path": ngrams_path,
                   "checkpoint_path": checkpoint_path,
                   "batch_size": cfg["training"]["batch_size"],                 # 16..128
                   "epochs": cfg["training"]["epochs"],                         # 10..100
                   "device": device,                                            # cpu, cuda, mps

                   "embed_latent_dimention": cfg["model"]["embed_latent_dim"],
                   "lstm_latent_dimention": cfg["model"]["lstm_latent_dim"],    # 64..512
                   "bi_lstm": cfg["model"]["bi_lstm"],
                   "dropout_rate" : cfg["model"]["dropout_rate"],               # 0.1..0.5
                   "n_grams_size": cfg["model"]["ngrams_size"],
                   "n_grams_shifting": cfg["model"]["ngrams_shifting"],
                   "optimizer_type": cfg["model"]["optim"],                     # SGD, Adam
                   "learning_rate": cfg["model"]["lr"],                         # SGD: 0.01..0.001, Adam: 0.001..0.0001 -> 0.0005
                   "beta1": cfg["model"]["beta1"],
                   "beta2": cfg["model"]["beta2"],
                   "weight_decay": cfg["model"]["weight_decay"],                # 0.01..0.0001
                  }
    torch.save({"hyperparams": hyperparams}, os.path.join(utilities_path,f"hyper_params_{HYPERPARMS_ID}.params"))
    print("- Hyper-Parameters Saved")
else:
    hyperparams = torch.load(os.path.join(utilities_path, f"hyper_params_{HYPERPARMS_ID}.params"))["hyperparams"]
    pprint(hyperparams)
    print(f"- Hyper-Parameters Correctly Loaded from hyper_params_{HYPERPARMS_ID}.params")



# Loading the data
json_data_loaded = load_json_data(train_set_path)
idx_prova = random.randint(0, len(json_data_loaded))
# print("Keys: ", json_data_loaded[idx_prova].keys())
# print("Index: ", json_data_loaded[idx_prova]["idx"])
# print("Tokens: ", json_data_loaded[idx_prova]["tokens"])
# print("Labels: ", json_data_loaded[idx_prova]["labels"])
    # BIO format example data
# print_BIO_example(json_data_loaded, n_examples=1)

    # Sentence Dataset Creation
training_data = Sentences_Dataset(raw_data=json_data_loaded,
                                  params=hyperparams,
                                  n_grams=True, is_training=True)
training_data.elements_to_be_removed(min_length=0, max_length=training_data.max_samp_length)
training_data.remove_elements_from_list()
print("- Training Data Loaded")



# Vocabulary and GloVe Pre-Trained Embeddings Integration
    # GloVe txt file processing
READ_GLOVE = cfg["project"]["glove"]["read"]
if READ_GLOVE == True:
  RECREATE_GLOVE = cfg["project"]["glove"]["recreate"]
  if RECREATE_GLOVE == True:
    glove_word_dict = read_glove_embeddings(glove_pretrained_txt)
    glove_words = list(glove_word_dict.keys())
    glove_name = f"glove_{glove_size}B_{hyperparams['embed_latent_dimention']}.glove"
    torch.save({"dict": glove_word_dict,
                "words": glove_words}, os.path.join(utilities_path,glove_name))
    print(f"- GloVe Processing saved into file {glove_name}")
  else:
      glove_name = f"glove_{glove_size}B_{hyperparams['embed_latent_dimention']}.glove"
      glove_word_dict = torch.load(os.path.join(utilities_path, glove_name))["dict"]
      glove_words = torch.load(os.path.join(utilities_path, glove_name))["words"]
      print(f"- GloVe Loaded from file {glove_name} --")
else:
  print("- GloVe Reading Skipped")
# check_out_words(training_data.data, glove_words)
    
    # Vocabulary Creation
RECREATE_VOCAB = cfg["project"]["vocab"]["recreate"]
ID_VOCAB = cfg["project"]["vocab"]["id"]
if RECREATE_VOCAB == True:
    print("- Vocabularies Creation")
    word2idx = word_to_idx(training_data.data, glove_words)
    idx2word = idx_to_word(word2idx)

    label2idx = {"O": 0,
                 "B-SENTIMENT": 1,
                 "B-CHANGE": 2,
                 "B-ACTION": 3,
                 "B-SCENARIO": 4,
                 "B-POSSESSION": 5,
                 "I-SENTIMENT": 6,
                 "I-CHANGE": 7,
                 "I-ACTION": 8,
                 "I-SCENARIO": 9,
                 "I-POSSESSION": 10,
                 "<pad>": 11}
    idx2label = idx_to_label(label2idx)

    VOCABULARIES = {"word2idx": word2idx, "label2idx": label2idx,
                    "idx2word": idx2word, "idx2label": idx2label}
    torch.save({"vocabularies": VOCABULARIES}, os.path.join(vocabulary_path, f"vocabulary_{ID_VOCAB}.vocab"))
    print("- Vocabularies Saved")
else:
    vocabulary_name = f"vocabulary_{ID_VOCAB}.vocab"
    VOCABULARIES = torch.load(os.path.join(vocabulary_path, vocabulary_name))["vocabularies"]
    print(f"- Vocabularies Correctly Loaded from {vocabulary_name}")

    # GloVe Integration
RECREATE_GLOVE_EMBEDS = cfg["project"]["glove"]["recreate_embed"]
# This piece of code needs to read glove.txt file first. Otherwise direcly load the embedding created
if RECREATE_GLOVE_EMBEDS == True:
    glove_pretrained_embedding = retrieve_glove_myDictionary(glove_word_dict,
                                                             glove_words=glove_words,
                                                             my_dict=VOCABULARIES['idx2word'],
                                                             max_length=hyperparams["embed_latent_dimention"])
    tensor_glove_pretrained = []
    for key, val in glove_pretrained_embedding.items():
        tensor_glove_pretrained.append(val)
    tensor_glove_pretrained = np.array(tensor_glove_pretrained, dtype=float)
    tensor_glove_pretrained = torch.from_numpy(tensor_glove_pretrained)
    print("- Tensor Word Embedding Shape: ", tensor_glove_pretrained.shape)
    print("- Tensor Word Embedding Example Shape: ", tensor_glove_pretrained[0].shape)
    print("- Tensor Word Embedding Example Type: ", tensor_glove_pretrained[0].dtype)
    glove_embed_name = f"glove_{glove_size}B_{hyperparams['embed_latent_dimention']}.embed"
    torch.save({"embed": tensor_glove_pretrained}, os.path.join(utilities_path, glove_embed_name))
    print(f"- GloVe PreTrained Embeddings saved into file {glove_embed_name}")
else:
    glove_embed_name = f"glove_{glove_size}B_{hyperparams['embed_latent_dimention']}.embed"
    tensor_glove_pretrained = torch.load(os.path.join(utilities_path, glove_embed_name))["embed"]
    print(f"- GloVe PreTrained Embeddings loaded from file {glove_embed_name}")



# N-Grams Creation
RANDOMIZE_DATA = cfg["project"]["ngrams"]["randomize_data"]
    # N-Grams Creation or Loading
REDUCED_NGRAMS = cfg["project"]["ngrams"]["reduced_ngrams"]
RECREATE_NGRAMS = cfg["project"]["ngrams"]["recreate_ngrams"]
if RECREATE_NGRAMS == True:
    training_data.n_grams_creation(params=hyperparams, save_result=True, reduced=REDUCED_NGRAMS)
    print("- Len N-Grammed Dataset: ", len(training_data.data))
else:
    file_name = f"{hyperparams['n_grams_size']}_gram_{hyperparams['n_grams_shifting']}.ngram" if REDUCED_NGRAMS==True else f"{hyperparams['n_grams_size']}_gram_{hyperparams['n_grams_shifting']}.engram"
    training_data.n_grams_loading(n_grams_path=hyperparams["n_grams_path"], file=file_name)
    print("- Selected N-Grams File: ", file_name)
    # Sentences Encoding (with "Padding")
training_data.encode_data(word2id=VOCABULARIES['word2idx'], label2id=VOCABULARIES["label2idx"],
                          max_length=training_data.max_samp_length)
print("- Shape: ", training_data.encoded_data.shape)
    # Add UNK Randomness
if RANDOMIZE_DATA == True:
    n_random = int(0.33*len(training_data.data))
    print("- N° of senteces to be randomized: ", n_random)
    training_data.randomize_encoded_n_grams(unk_token=VOCABULARIES["word2idx"]["<unk>"],
                                            n=n_random)



## Training Procedure
    # Data Labels Distribution and Class Weigths Creation
print("\n-- Class weights creation --")
n_samples, labels_frequency, scaled_labels_frequency = label_distribution(training_data.encoded_data.tolist())
classes_weigth = classes_weigths_creation(n_samples, VOCABULARIES["idx2label"], scaled_labels_frequency)
    # DataLoader
print("\n-- Data Batching --")
training_data_length = len(training_data.encoded_data)
print("- Training Dataset Length: ", training_data_length)
training_dataset = DataLoader(training_data.encoded_data, batch_size=hyperparams["batch_size"], shuffle=True)
    # Training Loop: start the Training Process
model_trainer = LSTM_Trainer(hyperparams=hyperparams,
                             dictionaries=VOCABULARIES,
                             load_checkpoint=cfg["training"]["resume"],
                             embed_state=tensor_glove_pretrained,
                             labels_weigths=classes_weigth,
                             dropout=True)
print("\n-- Training Loop --")
model_trainer.train(hyperparams, training_dataset)