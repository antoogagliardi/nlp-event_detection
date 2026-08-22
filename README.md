# NLP - Event Detection (ED)

A PyTorch-based sequence-labeling system that detects **events** inside natural language sentences, tagging each token with a BIO-style event category (`ACTION`, `CHANGE`, `SCENARIO`, `POSSESSION`, `SENTIMENT`).

## Introduction

Event detection is the task of identifying, within a sentence, the spans of text that describe something *happening*, an action taken, a change of state, a scenario being set, a possession changing hands or a sentiment being expressed. This project is set as a token classification problem: every word in a sentence is labeled with a `B-`/`I-`/`O` tag (BIO scheme) marking whether it begins, continues or falls outside an event span.

The model used is a bidirectional LSTM on top of word embeddings (optionally initialized from pre-trained GloVe vectors), trained on n-gram windows of tokenized labeled sentences.
<!-- Once trained, it's packaged behind a small Flask API and shipped as a Docker image so it can be evaluated or queried without any local Python setup. -->

# 📋 Table of Contents
 
* [Overview](#-overview)
* [Repository Structure](#-repository-structure)
* [Data](#-data)
* [Requirements](#-requirements)
* [Training](#-training)
* [Usage](#-usage)
* [References](#-references)

## 🔎 Overview
 
At its core, the model is an **embedding → BiLSTM → linear classifier** stack (`ED_Model` in `code/src/model.py`): each token is embedded, the sentence is encoded by a bidirectional LSTM, and a dense layer with softmax predicts a label distribution per token. Word embeddings can be initialized randomly or from pre-trained GloVe vectors.
 
Sentences are tagged token-by-token using the **BIO scheme**, with five event categories plus the "outside" tag:
 
- `O` — not part of an event
- `B-ACTION` / `I-ACTION` — an action being taken
- `B-CHANGE` / `I-CHANGE` — a change of state
- `B-SCENARIO` / `I-SCENARIO` — a scenario or setting being established
- `B-POSSESSION` / `I-POSSESSION` — a possession changing hands
- `B-SENTIMENT` / `I-SENTIMENT` — a sentiment being expressed

End-to-end, the project moves through four phases:
 
1. **Preprocess** raw JSONL sentences into padded, encoded n-gram windows (`code/src/data.py`).
2. **Train** the BiLSTM tagger with class-weighted cross-entropy loss, logging metrics and checkpoints every epoch (`code/train.py`).
3. **Evaluate** checkpoints with `seqeval` (precision, recall, F1, classification report) and a confusion matrix (`code/test_notebook.ipynb`, `code/src/plot.py`).
4. **Serve** the trained model behind a Flask API, containerized with Docker for reproducible, dependency-free evaluation (`docker/`, `Dockerfile`, `test.sh`).
 
 
## 🗂 Repository Structure
 
```
nlp-event_detection/
├── Dockerfile                    # Builds the serving image (copies model/, code/src, docker/)
├── report.pdf
├── requirements.txt              # Python dependencies
├── test.sh                       # Build → run → evaluate → teardown the Docker container
├── code/
│   ├── train.py                  # Training entry point
│   ├── test_notebook.ipynb       # Run evaluation/plots
│   ├── configs/
│   │   └── config.yaml           # Hyper-parameters
│   └── src/
│       ├── data.py
│       ├── model.py              # ED_Model (BiLSTM)
│       ├── plot.py
│       └── utils.py              # Config/data loaders, vocab builders, GloVe helpers
├── data/
│   ├── README.md
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── train.jsonl
├── docker/
│   ├── app.py
│   ├── evaluate.py               # Client script: sends a JSONL test set to the running API
│   ├── model.py
│   ├── simple_test.py
│   └── src/
│       └── implementation.py     # TestModel: loads checkpoint + serves predictions
└── logs/
    ├── server.stdout              # Container stdout (populated by test.sh)
    └── server.stderr              # Container stderr (populated by test.sh)
```
 
## 📊 Data
 
The model expects data in **JSON Lines** format, one sentence per line, with tokens aligned to labels:
 
```json
{"idx": 0, "tokens": ["Even", "the", "smallest", "person", "can", "change", "the", "future", "."], "labels": ["O", "O", "O", "O", "O", "B-ACTION", "O", "O", "O"]}
```

**Labels used (BIO scheme):**
`O`, `B-ACTION`/`I-ACTION`, `B-CHANGE`/`I-CHANGE`, `B-SCENARIO`/`I-SCENARIO`, `B-POSSESSION`/`I-POSSESSION`, `B-SENTIMENT`/`I-SENTIMENT`
 
During preprocessing, sentences are chunked into sliding n-gram windows (size and stride configurable), and a vocabulary (`word2idx` / `label2idx`) plus the resulting encoded tensors are cached to disk so they don't need to be regenerated on every run.
 
## 🛠️ Requirements

* Ubuntu distribution: either 20.04 or the current LTS (22.04) are perfectly fine.
* [Conda](https://docs.conda.io/projects/conda/en/latest/index.html), a package and environment management system particularly used for Python in the ML community.

### Setup Environment

To evaluate the final model it will be used Docker to remove any issue pertaining the code runnability. To run *test.sh*, we need to perform two additional steps:

* Install Docker
* Setup a client

`test.sh` essentially setups a server exposing the model through a REST API and then queries this server, evaluating it.

#### 1. Install Docker

```bash
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

> ⚠️ Unfortunately, for the latter command to have effect, you need to **reboot** your Ubuntu OS and re-login. **Do it** before proceeding.

#### 2. Setup Client

The model will be exposed through a REST server, in order to call it during the evaluation we need a client. The client is written in the evaluation script and it needs some dependencies to run: Use conda to create the environment for the client.<br>

```bash
conda create -n nlp-event_detection python=3.9
conda activate nlp-event_detection
pip install -r requirements.txt
```

 
## 🏋️ Training
 
Configuration lives in `code/configs/config.yaml` and covers the whole pipeline: device, GloVe/vocab/n-gram caching flags, model hyperparameters, and training settings (epochs, batch size, learning rate, checkpoint resuming).
 
1. Adjust `code/configs/config.yaml` (device, data paths, hyperparameters) as needed.
2. Run:
```bash
   cd code
   python train.py
```
 
This will:
- Build (or load) the vocabulary and, optionally, the GloVe embedding matrix
- Build (or load) the n-gram encoded dataset
- Train the `ED_Model` (embedding → BiLSTM → linear classifier → softmax) with class-weighted cross-entropy loss
- Log per-step and per-epoch metrics to a CSV file and save a checkpoint (`model_<epoch>.pt`) after every epoch. Training can be resumed from the last checkpoint by setting `training.resume: True` in the config.
- Model evaluation (F1, precision, recall, classification report, confusion matrix) can be interactively run from `code/test_notebook.ipynb`.
 
## 🚀 Usage
 
The trained model is served through a lightweight Flask API packaged in a Docker image.
 
**Prerequisites:** place your trained checkpoint (`model.pt`), hyperparameters (`hyper_params.params`), and vocabulary (`vocabulary.vocab`) files inside a `model/` directory at the project root — the Dockerfile copies this folder into the image.
 
### Running the inference server (Docker)
The `docker/` folder contains a self-contained Flask service that loads a trained checkpoint and serves predictions over HTTP. <br>

*test.sh* is a simple bash script. It automates the full cycle — build the image, start the container, run `docker/evaluate.py` against a JSONL test file, print accuracy/F1 metrics, then stop and remove the container (dumping logs to `logs/server.stdout` and `logs/server.stderr`) <br>
To run it:

```bash
# Build and run the server, then evaluate it against a test file
conda activate nlp-event_detection
bash test.sh data/test.jsonl
```
 
> ⚠️ Actually, you can replace *data/test.jsonl* to point to a different file, as far as the target file has the same format.

 
## 📚 References
 
- Jason P.C. Chiu and Eric Nichols. (2016). [Named entity
recognition with bidirectional LSTM-CNNs.]()

- Jeffrey Pennington, Richard Socher, and Christopher
Manning. (2014). [GloVe: Global vectors for word
representation.]()

- Lance Ramshaw and Mitch Marcus. (1995). [Text chunk-
ing using transformation-based learning.]()

- Hasim Sak, Andrew Senior, and Françoise Beaufays.
(2014). [Long short-term memory based recurrent neu-
ral network architectures for large vocabulary speech
recognition.]()
 

## 👤 Author

**Antonio Gagliardi**  
Email: [gaglia.anto95@gmail.com](mailto:gaglia.anto95@gmail.com)

