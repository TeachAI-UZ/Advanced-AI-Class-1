   ##### -- Made some changes on on lines 168 ~ 172 to clean the old files if the code is rerun.   ####


import torch
import os
from tqdm import tqdm
import numpy as np
from utils import plot_loss_acc

#torch.cuda.empty_cache()


def compute_accuracy(y_pred, y):

    '''
    
    Gets y_pred  &  y  and computes the accuracy.
    
    Arguments:
        y_pred - predicted label
        y      - ground truth label
        
    '''
    
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    accuracy = correct.float() / y.shape[0]
    
    return accuracy



def train(model, tr_dl, val_dl, criterion, optimizer, epochs,
        batch_size, lr, device, log_dir):
    
    '''
    
    Gets a model, train dataloader, validation dataloader,loss_function, optimizer, 
    number of epochs, batch size, device type;
    Trains the model and plots LOSS and ACCURACY curves after training.
    
    Arguments:
    
        model      - a trained model;
        tr_dl      - train dataloader;
        val_dl     - validation dataloader;
        criterion  - loss function;
        optimizer  - optimizer type;
        epochs     - number of epoch to train the model;
        device     - device type;
        batch size - amount of data dataloader gets
        log_dir    - model DIRECTORY to save the results
    
    '''

    
    max_val_acc = 0.0
    
    train_accs = []
    train_losses = []
    
    val_accs = []
    val_losses = []
    
    bestEpoch = 0
    
    print('--------------------------------------------------------------')

    # Define your execution device
    device_name = torch.cuda.get_device_name(device)
    print(f"The model will be running on {device_name} device\n")

    # Move the model to gpu
    model.to(device)

    # Start, loop along epochs to do the training
    for i in range(epochs):
        
        print(f'EPOCH {i}')
        
        # Set training loss and accuracy
        train_accuracy = 0.0
        train_loss = 0.0
        model.train()  #maybe no need?
        iteration = 1
        
        print('\nTRAINING')
        
        # Get through the training dataloader
        for _, batch in tqdm(enumerate(tr_dl, 0)):
            
            print('\rEpoch[' + str(i) + '/' + str(epochs) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(tr_dl)), end='')
            iteration += 1
            
            # Get the inputs and move them to device
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Predict classes using images from the training dataloader
            predictions = model(images)
            # Compute the loss based on model predictions and real labels
            loss = criterion(predictions, labels)
            # Backpropagate the loss
            loss.backward()
             # Adjust parameters based on the calculated gradients
            optimizer.step()
            #Calculate the train accuracy
            train_accuracy += compute_accuracy(predictions, labels).item()
            train_loss += loss.item()   # extract the loss value
        
        
        # Validation loop
        val_accuracy = 0.0
        val_loss = 0.0

         # Change to evaluation mode
        model.eval()
        iteration = 1

        print('')
        print('\nVALIDATION')

        # Conduct validation without gradients
        with torch.no_grad():
            for _, data in enumerate(val_dl):

                # Get the data and gt 
                images, labels = data
                
                print('\rEpoch[' + str(i) + '/' + str(epochs) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(val_dl)), end='')
                iteration += 1
                
                # Move the data and gt to the gpu
                images, labels = images.to(device), labels.to(device)
                
                # Get the model predictions
                predictions = model(images)
                
                loss = criterion(predictions, labels)
                # Get the validation accuracy and loss
                val_accuracy += compute_accuracy(predictions, labels).item()
                val_loss += loss.item()
            
            # Compute the accuracy and loss over all train/validation images
            # Save accuracy and loss values
            train_accs.append(train_accuracy / len(tr_dl))
            val_accs.append(val_accuracy / len(val_dl))
            train_losses.append(train_loss / len(tr_dl))
            val_losses.append(val_loss / len(val_dl))
            
            print('\n')
            print(f'- Train Acc: {(train_accuracy / len(tr_dl))*100:.2f}%')
            print(f'- Val Acc: {(val_accuracy / len(val_dl))*100:.2f}%')
            print(f'- Train Loss: {train_loss / len(tr_dl):.3f}')
            print(f'- Val Loss: {val_loss / len(val_dl):.3f}')
        
            
        # # Save the model every 10 epochs
        # if i % 10 == 0:
        #     torch.save(model.state_dict(), checkpoints_path + "/checkpoint_" + str(i) + ".pth")
            
        # Save the best model
        if (val_accuracy / len(val_dl)) > max_val_acc:
            
            # Remove the previous best model and save the current best model if code is rerun
            if i == 0:
                for file in os.listdir(log_dir):
                  os.remove(log_dir + '/' + file)
        
                torch.save(model.state_dict(), log_dir + '/' + "checkpoint_" + str(i) + "_best.pth")
            else:
                os.remove(log_dir + '/'  + "checkpoint_" + str(bestEpoch) + "_best.pth")
                torch.save(model.state_dict(), log_dir + '/' + "checkpoint_" + str(i) + "_best.pth")
                
                # Remove previous loss and accuracy files to update the txt files with the best epoch
                os.remove(log_dir + '/train_accuracies_epochs' + str(epochs) + '_bs' + str(batch_size) +
                            '_lr' + str(lr) + '_bestEpoch' + str(bestEpoch) + '.txt')
                os.remove(log_dir + '/val_accuracies_epochs' + str(epochs) + '_bs' + str(batch_size) +
                            '_lr' + str(lr) + '_bestEpoch' + str(bestEpoch) + '.txt')
                os.remove(log_dir + '/train_losses_epochs' + str(epochs) + '_bs' + str(batch_size) +
                            '_lr' + str(lr) + '_bestEpoch' + str(bestEpoch) + '.txt')
                os.remove(log_dir + '/val_losses_epochs' + str(epochs) + '_bs' + str(batch_size) +
                            '_lr' + str(lr) + '_bestEpoch' + str(bestEpoch) + '.txt')
            
            print(f'\nAccuracy increased ({max_val_acc*100:.6f}% ---> {(val_accuracy / len(val_dl))*100:.6f}%) \nModel saved')
            
            # Update parameters with the new best model
            max_val_acc = val_accuracy / len(val_dl)
            bestEpoch = i
            
            
        print("--------------------------------------------------------------")
        
        
        # Save losses and accuracies
        np.savetxt(log_dir + '/train_accuracies_epochs' + str(epochs) + '_bs' + str(batch_size) +
                    '_lr' + str(lr) + '_bestEpoch' + str(bestEpoch) + '.txt', np.array(train_accs))
        np.savetxt(log_dir + '/val_accuracies_epochs' + str(epochs) + '_bs' + str(batch_size) +
                    '_lr' + str(lr) + '_bestEpoch' + str(bestEpoch) + '.txt', np.array(val_accs))
        np.savetxt(log_dir + '/train_losses_epochs' + str(epochs) + '_bs' + str(batch_size) +
                    '_lr' + str(lr) + '_bestEpoch' + str(bestEpoch) + '.txt', np.array(train_losses))
        np.savetxt(log_dir + '/val_losses_epochs' + str(epochs) + '_bs' + str(batch_size) +
                    '_lr' + str(lr) + '_bestEpoch' + str(bestEpoch) + '.txt', np.array(val_losses))
        

    # Plot losses and accuracy curves
    plot_loss_acc(log_dir)
    

