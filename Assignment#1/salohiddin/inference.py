### - Made sliight changes on lines 35 ~ 37 because of importing 'glob'.

import torch, timm, re, glob
from  tqdm import tqdm
from utils import plot_confusion_matrix

def inference(model_name, num_classes, device, dl, cls_names):
    
    '''
    Gets a model name, number of classes for the dataset, model directory, device type, dataloader and class names;
    performs inference and plots the confusion matrix on performance.
    
    Arguments:
    model_name  - model name for training;
    num_classes - number of classes for the dataset;
    device      - device type;
    dl          - dataloader.
    
    '''
    
    # Create lists for predictions, ground truths, and images
    predictions, gts, images = [], [], []
    
    # Create a model
    model = timm.create_model(model_name, pretrained = True, num_classes = num_classes)
    
    # Move the model to gpu
    model.to(device)
    
    # Load checkpoint from the model directory
    # Already inside the model_dir because of "plot_loss_acc()" call in the train.py
    dir_files = glob.glob('*.txt')  
    train_acc_file = dir_files[0]
    bestEpoch = re.findall('[0-9]+', train_acc_file)
    bestEpoch = int(bestEpoch[-1])

    model.load_state_dict(torch.load(f"./checkpoint_{bestEpoch}_best.pth"))
    print("Model checkpoint loaded successfully!")
    
    # Set initial correct cases and total samples
    correct, total = 0, 0
    
    # Go through the dataloader
    for _, batch in tqdm(enumerate(dl, 0)):
    #for idx, batch in tqdm(enumerate(dl)):
        
        # Get images and gt labels
        ims, lbls = batch
        
        # Get predictions
        ims = ims.to(device)
        preds = model(ims)
        images.extend(ims)
        
        # Get classes with max values
        _, predicted = torch.max(preds.data, 1)
        
        # Add to predictions list
        predictions.extend(predicted)
        
        # Add gt to gts list
        gts.extend(lbls.to(device))
        
        # Add batch size to total number of samples
        total += lbls.size(0)
        
        # Get correct predictions
        correct += (predicted == lbls.to(device)).sum().item()        
    test_acc = 100 * correct // total    
    print(f'Accuracy of the network on the {total} test images: {test_acc} %')  
    
    # Return model, predictions, ground truths, and images
    #return model, torch.stack(predictions), torch.stack(gts), torch.stack(images)
    
    plot_confusion_matrix(gts, predictions, cls_names, test_acc)
