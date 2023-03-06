################## Comments on utils.py ##################
    # 1. There is a lot of repeated code in plot_loss_acc function. Can you make it more efficient?;
     # -- Code is updated based on feedback.

# Import libraries
import os, glob, re, torch
import numpy as np
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns

def create_model_dir(log_dir):

    '''
    
    This function gets log_dir, creates DIRECTORY (if not available) where model's best parameters and outputs are saved,
    and return log_dir itself

    Arguments:
    
       log_dir - model DIRECTORY

    '''

    os.makedirs(log_dir, exist_ok=True)

    return log_dir

def plot_loss_acc(log_dir):
    
    '''
    
    This function gets log_dir - the DIRECTORY where model's best parameters and output data are saved,
    and plots LOSS and ACCURACY curves.

    Arguments:
    
       log_dir - model DIRECTORY

    '''
    os.chdir(log_dir)
    dir_files = glob.glob('*.txt')

    train_acc_file, train_loss_file = dir_files[0], dir_files[1]
    val_acc_file, val_loss_file = dir_files[2], dir_files[3]
    
    train_accs = np.loadtxt(train_acc_file)
    train_losses = np.loadtxt(train_loss_file)
    val_accs = np.loadtxt(val_acc_file)
    val_losses = np.loadtxt(val_loss_file)

    bestEpoch = re.findall('[0-9]+', train_acc_file)
    bestEpoch=int(bestEpoch[-1])
    
    epochs = np.arange(train_losses.shape[0])
    
    plt.figure()
    plt.grid(which='both', axis='both')

    plt.plot(epochs, train_losses, label="Training loss", c='b')
    plt.plot(epochs, val_losses, label="Validation loss", c='r')
    plt.plot(bestEpoch, val_losses[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+.01, val_losses[bestEpoch]+.01, str(bestEpoch) + ' - ' + str(round(val_losses[bestEpoch], 3)), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss along epochs')
    plt.legend()
    plt.draw()
    plt.savefig('./1_loss.png')
    
    plt.figure()
    plt.grid(which='both', axis='both')
    plt.plot(epochs, train_accs, label="Training accuracy", c='b')
    plt.plot(epochs, val_accs, label="Validation accuracy", c='r')
    plt.plot(bestEpoch, val_accs[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+.001, val_accs[bestEpoch]+.001, str(bestEpoch) + ' - ' + str(round(val_accs[bestEpoch], 3)), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy along epochs')
    plt.legend()
    plt.draw()
    plt.savefig('./2_accuracy.png')

def plot_confusion_matrix(gts, pred, cls_names, test_acc): # log_dir,
    
    '''
    
    This function gets GROUND TRUTHS labels and PREDICTED labels, class names and log_dir - the DIRECTORY 
    where model's best parameters and output data are saved, and plots CONFUSION MATRIX

    Arguments:
    
           gts       - GROUND TRUTS labels
           pred      - PREDICTED labels
           cls_names - class names in the dataset
           log_dir   - model DIRECTORY

    '''

    y_true = torch.tensor(gts).detach().cpu().numpy()
    y_pred = torch.tensor(pred).detach().cpu().numpy()

    # Define class names
    class_names =  cls_names
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    # Plot the Confusion Matrix (3 color types are available)
    plt.figure(figsize=(12,8))
    sns.heatmap(cm, fmt='.2g', annot=True, cmap="crest", xticklabels=class_names, yticklabels=class_names) #2nd type

    # Add labels and title
    plt.title(f"Confusion Matrix: Test acc = {test_acc} %")
    plt.xlabel('Predicted label')
    plt.ylabel('True Label')

    # Rotate tick labels and set alignment
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Save and show plot
    plt.savefig('./3_cmatrix.png')
