################## Comments on utils.py ##################
    # 1. There is a lot of repeated code in plot_loss_acc function. Can you make it more efficient?;

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix 
import seaborn as sns





def create_model_dir(log_dir):

    '''
    
    Gets log_dir creates DIRECTORY (if not available) where model's best parameters and outputs are saved,
    and return log_dir itself

    Arguments:
       log_dir - model DIRECTORY

    '''

    os.makedirs(log_dir, exist_ok=True)

    return log_dir



def plot_loss_acc(log_dir):
    
    '''
    
    Gets log_dir - the DIRECTORY where model's best parameters and output data are saved,
    and plots LOSS and ACCURACY curves.

    Arguments:
       log_dir - model DIRECTORY

    '''

    
    network_files = os.listdir(log_dir)
    
    train_acc_file = [string for string in network_files if 'train_accuracies' in string]
    train_accs = np.loadtxt(log_dir + '/' + train_acc_file[0])
    
    validation_acc_file = [string for string in network_files if 'val_accuracies' in string]
    validation_accs = np.loadtxt(log_dir + '/' + validation_acc_file[0])
    
    train_loss_file = [string for string in network_files if 'train_losses' in string]
    train_losses = np.loadtxt(log_dir + '/' + train_loss_file[0])
    
    validation_loss_file = [string for string in network_files if 'val_losses' in string]
    validation_losses = np.loadtxt(log_dir + '/' + validation_loss_file[0])
    
    bestEpoch = validation_acc_file[0].split('_')
    bestEpoch = bestEpoch[-1]
    bestEpoch = bestEpoch.split('.')
    bestEpoch = bestEpoch[0]
    bestEpoch = int(re.search(r'\d+', bestEpoch).group())
    
    epochs = np.arange(train_losses.shape[0])
    
    plt.figure()
    plt.grid(which='both', axis='both')

    plt.plot(epochs, train_losses, label="Training loss", c='b')
    plt.plot(epochs, validation_losses, label="Validation loss", c='r')
    plt.plot(bestEpoch, validation_losses[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+.01, validation_losses[bestEpoch]+.01, str(bestEpoch) + ' - ' + str(round(validation_losses[bestEpoch], 3)), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(log_dir + '/1_loss.png')
    
    plt.figure()
    plt.grid(which='both', axis='both')
    plt.plot(epochs, train_accs, label="Training accuracy", c='b')
    plt.plot(epochs, validation_accs, label="Validation accuracy", c='r')
    plt.plot(bestEpoch, validation_accs[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+.001, validation_accs[bestEpoch]+.001, str(bestEpoch) + ' - ' + str(round(validation_accs[bestEpoch], 3)), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(log_dir + '/2_accuracy.png')
    
    # plt.show()


def plot_confusion_matrix(gts, pred, cls_names, log_dir, test_acc):
    
    '''
    
    Gets GROUND TRUTHS labels and PREDICTED labels, class names and log_dir - the DIRECTORY 
    where model's best parameters and output data are saved, and plots CONFUSION MATRIX

    Arguments:
       gts - GROUND TRUTS labels
       pred - PREDICTED labels
       cls_names - class names in the dataset
       log_dir - model DIRECTORY

    '''

    #y_true, y_pred = gts.cpu(), pred.cpu()   ??
    y_true = torch.tensor(gts).detach().cpu().numpy()
    y_pred = torch.tensor(pred).detach().cpu().numpy()

    # Define class names
    class_names =  cls_names
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    # Plot the Confusion Matrix (3 color types are available)
    plt.figure(figsize=(12,8))
    #sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names)  #1st type
    sns.heatmap(cm, fmt='.2g', annot=True, cmap="crest", xticklabels=class_names, yticklabels=class_names) #2nd type
    #sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, cmap=sns.cubehelix_palette(as_cmap=True)) #3rd type

    # Add labels and title
    plt.title(f"Confusion Matrix: Test acc = {test_acc} %")
    plt.xlabel('Predicted label')
    plt.ylabel('True Label')

    # Rotate tick labels and set alignment
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Save and show plot
    plt.savefig(log_dir + '/3_cmatrix.png')
   # plt.show()
