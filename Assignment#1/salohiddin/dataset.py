from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch


def get_dl(root, batch_size, t):
    
    '''
    
    Gets a path to the data and returns class names, number of classes, train dataloader, and validation dataloader.
    
    Arguments:
        root - path to the images;
        bs - batch size of the dataloaders;
        t - transformations;
        
    Outputs:
    
        cls_names - names of the classes in the dataset;
        num_classes - number of the classes in the dataset;
        tr_dl - train dataloader;
        val_dl - validation dataloader;
        test_dl - test dataloader.
        
    '''
    
    # Get dataset from the directory
    ds = ImageFolder(root = root, transform = t)
    
    # Get length of the dataset
    ds_length = len(ds)
    
    # Split the dataset into train, validation and test datasets
    tr_ds_lenth = int(ds_length * 0.8)
    val_ds_lenth = int(ds_length * 0.1)
    test_ds_lenth = ds_length - tr_ds_lenth - val_ds_lenth

    tr_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [tr_ds_lenth, val_ds_lenth, test_ds_lenth])

    print(f"Number of train set images: {len(tr_ds)}")
    print(f"Number of validation set images: {len(val_ds)}")
    print(f"Number of test set images: {len(test_ds)}")
    
    # Get class names
    cls_names = list(ds.class_to_idx.keys())
    # Get total number of classes
    num_classes = len(cls_names)
    
    # Create train and validation dataloaders
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Return class names, total number of classes, train and validation dataloaders
    return cls_names, num_classes, tr_dl, val_dl, test_dl
