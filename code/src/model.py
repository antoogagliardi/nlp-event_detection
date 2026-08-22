import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

#-- Metrics
from seqeval.metrics import accuracy_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report
from seqeval.scheme import IOB1, IOB2


from src.utils import inspect_checkpoints
from src.plot import plot_confusion_matrix


## Entity-Detection Model
class ED_Model(nn.Module):
    def __init__(self, dictionaries:dict, hyperparams:dict, enable_dropout:bool=False):
        super(ED_Model,self).__init__()

        self.VOCABULARIES = dictionaries
        self.VOCABULARY_SIZE = len(dictionaries["word2idx"])                                    # ricavato dal len(word2idx) dictionary
        self.LABELS_SIZE = len(dictionaries["label2idx"])                                       # ricavato dal len(tag2idx) dictionary
        self.EMBEDDING_LATENT_DIM = hyperparams["embed_latent_dimention"]
        self.LSTM_LATENT_DIM = hyperparams["lstm_latent_dimention"]
        self.DROPOUT_RATE = None if enable_dropout == False else hyperparams["dropout_rate"]

         # WORD EMBEDDING MATRIX (LOOK-UP TABLE)
        self.word_embedding = nn.Embedding(num_embeddings=self.VOCABULARY_SIZE,
                                           embedding_dim=self.EMBEDDING_LATENT_DIM).to(device=hyperparams["device"])

        # LSTM
        self.lstm = torch.nn.LSTM(input_size=self.EMBEDDING_LATENT_DIM, hidden_size=self.LSTM_LATENT_DIM,
                                  num_layers=2, bias=True, batch_first=True,
                                  dropout=0 if enable_dropout==False else self.DROPOUT_RATE,
                                  bidirectional=False if hyperparams["bi_lstm"] is False else True).to(device=hyperparams["device"])
        if enable_dropout == True and self.DROPOUT_RATE != None:
            # DROPOUT LAYER
            self.dropout = nn.Dropout(p=self.DROPOUT_RATE).to(device=hyperparams["device"])

        # CLASSIFIER
        self.classifier = nn.Linear(in_features= self.LSTM_LATENT_DIM if hyperparams["bi_lstm"] is False else self.LSTM_LATENT_DIM*2,
                                    out_features=self.LABELS_SIZE).to(device=hyperparams["device"])

    def forward(self, input_sequence):                          # input_sequence.shape=[32, 156]
        # Embedding of the SENTENCES ONLY
        embeddings = self.word_embedding(input_sequence)        # embeddings.shape=[32, 156, 100]

        output_features, _ = self.lstm(embeddings)              # output_features.shape=[32, 156, 312]

        if self.DROPOUT_RATE != None:
            #-- Apply a Dropout Layer on the output features before passing them to the classifier
            output_features = self.dropout(output_features)

        #-- Dense Layer
        predicted_labels = self.classifier(output_features)     # predicted_labels.shape=[32, 156, 12]

        #-- Activation Function
        labels_score = F.softmax(predicted_labels, dim=-1)      # labels_score.shape=[32, 156, 12]

        return labels_score
    

## Model Trainer
class LSTM_Trainer:
    METRICS_COLUMNS = ["event", "split", "epoch", "step", "global_step", "loss", "accuracy"]
    # Usefull variables to resume training from checkpoints
    checkpoint_loaded = False
    last_epoch = None

    def __init__(self, hyperparams:dict, dictionaries:dict,
                 load_checkpoint:bool=False, embed_state=None,
                 labels_weigths=None, dropout=False):
        print("-- Model Inizialization --")
        # MODEL INSTATIATION
        self.ed_model = ED_Model(dictionaries=dictionaries, hyperparams=hyperparams, enable_dropout=dropout)
        if self.ed_model.DROPOUT_RATE != None:
            print(f"- Dropout is enabled in the model, Dropout Rate: {self.ed_model.DROPOUT_RATE}")

        # OPTIMIZERS CONFIGURATION
        self.model_opt = self.configure_optimizer(hyperparams, hyperparams["optimizer_type"])

        # CHECKPOINT LOADING ----------------------------------------------------
        if load_checkpoint == True:
            print("- Load Model from last training checkpoint")
            last_checkpoint, checkpoint_dict = self.load_checkpoint(hyperparams)

            # Load the parameters of the last training into the models ----------
            self.ed_model.load_state_dict(checkpoint_dict["model"]); print(" - Last Training Checkpoint Loaded")
            self.model_opt.load_state_dict(checkpoint_dict["model_opt"]); print(" - Optimizer State Loaded")
            self.ed_model.VOCABULARIES = checkpoint_dict["dictionaries"]; print(" - Model's Dictionaries Loaded")

            # Put the model into training mode
            self.ed_model.train()

            self.checkpoint_loaded = True
            self.last_epoch = last_checkpoint
            print("-- Last trained model has been loaded correctly --")
        else:
            self.last_epoch = 0
            if embed_state == None:
                self.ed_model.word_embedding.weight.data.uniform_(-1.0 / self.ed_model.VOCABULARY_SIZE,
                                                                  1.0 / self.ed_model.VOCABULARY_SIZE)
                print("- Weights Initiliazed from Normal Distribution")
            else:
                self.ed_model.word_embedding.weight.data.copy_(embed_state)
                print("- Weights Initiliazed from Pre-Trained Word Embedding")

            # Put the model into training mode
            self.ed_model.train()
            # -----------------------------------------------------------------------

        # LOSS CRITERION: Cross Entropy Loss Function: the problem is a Multi-Classification problem with C=11
        if labels_weigths != None:
            self.classes_weigths = labels_weigths
            self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.classes_weigths, ignore_index=11,
                                                          reduction="mean").to(device=hyperparams["device"])
        else:
            self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=11, reduction="mean").to(device=hyperparams["device"])

    def log_metrics(self, filepath, event, split, epoch=np.nan, step=np.nan, global_step=np.nan,
                    loss=np.nan, accuracy=np.nan, precision=np.nan,
                    recall=np.nan, f1=np.nan):
        """Appends one training/evaluation record to the shared CSV file."""
        metrics = pd.DataFrame([{
            "event": event,
            "split": split,
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }], columns=self.METRICS_COLUMNS)
        metrics.to_csv(filepath, mode="a", header=not os.path.isfile(filepath), index=False)

    def load_checkpoint(self, params:dict):
        last_checkpoint = inspect_checkpoints(params["checkpoint_path"])
        print(" - Last Checkpoint Loaded: ", last_checkpoint)
        checkpoint_path = os.path.join(params["checkpoint_path"], "model_{}.pt".format(last_checkpoint))
        if torch.cuda.is_available() == True:
            checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device(params["device"]), weights_only=False)
            print("  - There's a runtime with CUDA GPU in use")
        else:
            checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device(params["device"]), weights_only=False)
            print("  - There's no CUDA GPU available, Tensors would be loaded on: {}".format(params["device"]))

        # Return the number of last checkpoint and its "state dictionary"
        return last_checkpoint, checkpoint_dict

    def decode_output(self, data, id2label):
        decoded_data = []
        for row in data:
            y_true = []
            y_pred = []
            y_true.append(id2label[row[0]])
            y_pred.append(id2label[row[1]])
            decoded_data.append((y_true,y_pred))

        return decoded_data
    
    # OPTIMIZER CONFIGURATION: (SGD or Adam)
    def configure_optimizer(self, params, opt_type:str="SGD"):
        lr = params["learning_rate"]
        if opt_type == "SGD":
            model_opt = torch.optim.SGD(self.ed_model.parameters(), lr=lr)
        elif opt_type == "Adam":
            model_opt = torch.optim.Adam(list(self.ed_model.parameters()),
                                        lr=lr,
                                        eps=1e-8,
                                        weight_decay=params["weight_decay"],
                                        betas=(params["beta1"], params["beta2"])
                                        )
        return model_opt

    # TRAINING LOOP
    def train(self, params, training_set):
        if self.checkpoint_loaded == True:
            print("- Resuming the training process from Epoch {}".format(self.last_epoch+1))
            epochs = range(self.last_epoch+1,self.last_epoch+1+params["epochs"])
        else:
            epochs = range(params["epochs"])
        steps_per_epoch = 10 # len(training_set)                                 # Number of batches we have in one single epoch
        print("- N° of batches in one single epoch: ", steps_per_epoch)
        print("- N° of training epochs: ", params["epochs"])

        # Training Loop
        for epoch in epochs:
            accuracy_list = 0.0
            loss_list = 0.0
            with tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}" if not self.checkpoint_loaded else f"Epoch {epoch+1}") as pbar:
                for i, xt in zip(pbar, training_set):   # Iteration over all the batches
                    # torch.narrow: returns a narrowed version of the original tensor
                    #  (used to slice the tensors by defining the dim, start and length parameters)
                    sentences = torch.narrow(xt, 1, 0, 1)                                                   # sentences.shape=[32, 1, 156]
                    sentences = torch.reshape(sentences, shape=(sentences.shape[0], sentences.shape[2]))    # sentences.shape=[32, 156]

                    target = torch.narrow(xt, 1, 1, 1)                                                      # target.shape=[32, 1, 156]
                    target = torch.reshape(target, shape=(target.shape[0], target.shape[2]))                # target.shape=[32, 156]

                    # Load Sentences and Targets on the device
                    sentences = sentences.to(device=params["device"])
                    target = target.to(device=params["device"])
                    target = torch.flatten(target)                                                          # target.shape=[32*156]

                    # Zero the parameter gradients
                    self.model_opt.zero_grad()

                    # Calling the model
                    model_output = self.ed_model(sentences)                                                 # model_output.shape=[32, 156, 12]
                    model_output = model_output.view(-1, model_output.shape[-1])                            # model_output.shape=[32*156, 12]

                    # Loss calculation
                    #  - input.shape = [N, C, d]
                    #  - target.shape = [N, d] -> SCALAR
                    model_loss = self.cross_entropy_loss(model_output, target)                              # model_loss.shape = SCALAR

                    # BACKWARD PASS
                        # Backward Model Loss
                    model_loss.backward()       # model_loss.backward(retain_graph=True)
                        # Weights update
                    self.model_opt.step()

                    # ---------------- SPERIMENTALE METRICS  ----------------------------------------
                    _, top_label_indices = torch.max(model_output, -1)                                      # top_label_indices.shape=[32*156]
                    mask = target < 11                                                                      # mask.shape=[32*156], 11: is the id of "<pad>"
                    most_relevant = top_label_indices[mask]                                                 # variabile in base alla vera lunghezza della parola
                    
                    accuracy = ((most_relevant == target[mask]).sum()/most_relevant.shape[0]).item()
                    accuracy = np.round(accuracy, 6)
                    accuracy_list += accuracy

                    loss = model_loss.item()
                    loss = np.round(loss, 6)
                    loss_list += loss
                    # -------------------------------------------------------------------------------

                    global_step = epoch * steps_per_epoch + i + 1
                    self.log_metrics(
                        filepath=params.get("metrics_file", os.path.join(params["checkpoint_path"], "training_metrics.csv")),
                        event="step", split="train",
                        step=i + 1, global_step=global_step,
                        loss=loss, accuracy=accuracy,
                    )

                    pbar.set_postfix(LOSS=model_loss.cpu().detach().numpy().item())
                    pbar.update(0)
                epoch_loss = np.round(loss_list/steps_per_epoch, 6)
                epoch_accuracy = np.round(accuracy_list/steps_per_epoch, 6)
                self.log_metrics(
                    filepath=params.get("metrics_file", os.path.join(params["checkpoint_path"], "training_metrics.csv")),
                    event="epoch", split="train",
                    epoch=epoch if not self.checkpoint_loaded else epoch + 1,
                    loss=epoch_loss, accuracy=epoch_accuracy,
                )

                # Save model checkpoint
                print("Saving results of the training at epoch {}".format(epoch if not self.checkpoint_loaded else epoch+1))
                ckpt_name = f"model_{epoch}.pt" if not self.checkpoint_loaded else f"model_{self.last_epoch+1}.pt"
                torch.save({"model": self.ed_model.state_dict(),
                            "model_opt": self.model_opt.state_dict(),
                            "dictionaries": self.ed_model.VOCABULARIES}, os.path.join(params["checkpoint_path"], ckpt_name))
                if self.checkpoint_loaded == True:
                    self.last_epoch += 1
            print(f"Metrics Calculation for the Epoch {epoch} are:\
                  \n - Loss={epoch_loss}\
                  \n - Accuracy={epoch_accuracy}")

    # EVALUATION LOOP
    def evaluate(self, params, evaluation_set):
        cumm_accuracy = 0.0
        cumm_loss = 0.0
        total_predictions = []

        # Put the Model in EVALUATION MODEL
        self.ed_model.eval()

        steps_per_epoch = len(evaluation_set)
        with torch.no_grad(): # Deactivate Autograd
            with tqdm(range(steps_per_epoch)) as pbar:
                for i, xt in zip(pbar, evaluation_set): # Iteration over all the batches
                    sentences = torch.narrow(xt, 1, 0, 1)                                                   # sentences.shape=[32, 1, 156]
                    sentences = torch.reshape(sentences, shape=(sentences.shape[0], sentences.shape[2]))    # sentences.shape=[32, 156]

                    target = torch.narrow(xt, 1, 1, 1)                                                      # target.shape=[32, 1, 156]
                    target = torch.reshape(target, shape=(target.shape[0], target.shape[2]))                # target.shape=[32, 156]

                    # Load Sentences and Targets on the device
                    sentences = sentences.to(device=params["device"])
                    target = target.to(device=params["device"])

                    # Calling the Model to do its predictions
                    model_output = self.ed_model(sentences)                                                 # model_output.shape = [32, 156, 12]

                    model_output = model_output.view(-1, model_output.shape[-1])                            # model_output.shape=[32*156, 12]
                    target = torch.flatten(target)                                                          # target.shape=[32*156]

                    # Compute the Loss
                    loss = self.cross_entropy_loss(model_output, target)                                    # model_loss.shape = SCALAR or [32, 156]

                    # ------- METRICHE ---------------------------------------------------------
                    # Compute the Accuracy
                    _, top_label_indices = torch.max(model_output, -1)
                    mask = target < 11                                                                      # 11: is the id of "<pad>"
                    most_relevant = top_label_indices[mask]
                    accuracy = ((most_relevant == target[mask]).sum()/most_relevant.shape[0]).item()
                    cumm_accuracy += accuracy

                    loss = loss.item()
                    cumm_loss += loss
                    # ---------------------------------------------------------------------------

                    out = most_relevant.cpu().numpy()
                    ground_truth = target[mask].cpu().numpy()
                    data = []
                    for targ, pred in zip(out, ground_truth):
                        data.append([targ.tolist(),pred.tolist()])
                    decoded_out = self.decode_output(data,self.ed_model.VOCABULARIES["idx2label"])
                    total_predictions.append(decoded_out)
        total_target = []
        total_pred = []
        for rows in total_predictions:
            for row in rows:
                for targ, pred in zip(row[0], row[1]):
                    total_target.append(targ)
                    total_pred.append(pred)
        print("F1-Score: ", f1_score([total_target], [total_pred]))
        print("Precision: \n", precision_score([total_target], [total_pred]))
        print("Racall: \n", recall_score([total_target], [total_pred]))
        print("Classification Report: \n", classification_report([total_target], [total_pred],
                                                                 mode='strict', scheme=IOB2,
                                                                 digits=6))
        print("Final Loss: ", np.round(cumm_loss/steps_per_epoch, 6))
        print("Final Accuracy: ", np.round(cumm_accuracy/steps_per_epoch, 6))
        print("\n")
        plot_confusion_matrix(sum([total_target], []), sum([total_pred], []),
                              classes = self.ed_model.VOCABULARIES["idx2label"],
                              normalize=True,
                              title="Model Evaluation",
                              cmap=plt.cm.OrRd)