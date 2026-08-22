import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#-- SkLearn Confusion Matrix: visualize the training report information
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_training_results(res_file:str, type_res:str="loss", r_mode:str="epoch"):

    csv_file = pd.read_csv(filepath_or_buffer=res_file)

    data_plot = []
    for i in range(len(csv_file)):
        if r_mode == "epoch":
            if not np.isnan(csv_file.iloc[i]["epoch"]):
                if type_res=="loss":
                    data_plot.append(csv_file.iloc[i]["loss"])
                if type_res=="acc":
                    data_plot.append(csv_file.iloc[i]["accuracy"])
        if r_mode == "step":
            if np.isnan(csv_file.iloc[i]["epoch"]):
                if type_res=="loss":
                    data_plot.append(csv_file.iloc[i]["loss"])
                if type_res=="acc":
                    data_plot.append(csv_file.iloc[i]["accuracy"])

    legend_loc = "upper right"
    label = f"{type_res} {r_mode}"
    mycolor = "blue"
    plt.figure(figsize=(50, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.plot(data_plot, color=mycolor, linestyle="-", marker="o", label=label)
    plt.title(f"{label}")
    plt.xlabel("Training Epochs")
    plt.ylabel(f"{type_res}")
    plt.grid(True)
    plt.legend(loc=legend_loc)

    plt.show()

def plot_histogram(data):
    # Get the values from the dictionary
    keys = list(data.keys())
    values = list(data.values())
    # Plot the histogram
    # Create a figure and specify its size
    fig = plt.figure(figsize=(100, 100))
    plt.bar(keys, values, color="red", edgecolor='purple', width=0.5)

    # Add labels and title to the plot
    plt.xlabel('Length of a phrase')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.yticks(np.arange(1, max(values), 2))
    plt.xticks(np.arange(1, max(keys), 1))
    plt.grid(True)

    # Show the plot
    plt.show()

# Custom function to plot the Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = "Normalized Confusion Matrix"
        else:
            title = "Confusion Matrix"
    # Compute Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Use the labels that are in our dataset
    classes = unique_labels(y_true, y_pred)
    classes = classes.tolist()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm[np.isnan(cm)] = 0  # Handle NaNs (zero division)

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 9)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Main Loop over data dimensions
    fmt = '.2f' if normalize else 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]*100, fmt) + "%" if normalize else format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black")  # Set text color to black

    fig.tight_layout()
    return ax
