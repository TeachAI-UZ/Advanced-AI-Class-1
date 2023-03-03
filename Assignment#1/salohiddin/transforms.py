#### - Slightly orginized and added tfs.CenterCrop(), needed my actual data ###

from torchvision import transforms as tfs
#from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD



def get_transforms(train=False):
    
    '''
    Gets argment to distinguish train or validation transformations and return transforms.
    
    Arguments:
    train - train transformation if True, else validation transformations.
    '''
    
    # Train transformations
    t_tr = tfs.Compose([
                        tfs.CenterCrop((550, 480)),
                         #tfs.Resize((299, 299)),  #
                         tfs.Resize((224, 224)),
                       #tfs.RandomCrop((120, 120)),
                       tfs.RandomHorizontalFlip(p=0.3),
                       tfs.RandomRotation(degrees=15),
                       tfs.RandomVerticalFlip(p=0.3),
                        tfs.RandomAffine((30, 70), translate=(0.1, 0.3)),
                       #tfs.Grayscale(num_output_channels=3),
                       tfs.ToTensor(),
                    #    tfs.Normalize(mean=IMAGENET_DEFAULT_MEAN, 
                    #                  std=IMAGENET_DEFAULT_STD), 
                       tfs.Normalize(mean=IMAGENET_INCEPTION_MEAN,
                              std=IMAGENET_INCEPTION_STD)  
                         #tfs.Normalize(mean=[0.5, 0.5, 0.5],
                            #  std=[0.5, 0.5, 0.5]) 
                       ])
    
    # Validation and test transformations
    t_val = tfs.Compose([tfs.CenterCrop((550, 480)),
                         tfs.Resize((224,224)),
                         #tfs.Grayscale(num_output_channels=3),
                         tfs.ToTensor(),
                        #  tfs.Normalize(mean=IMAGENET_DEFAULT_MEAN, 
                        #              std=IMAGENET_DEFAULT_STD), 
                         tfs.Normalize(mean=IMAGENET_INCEPTION_MEAN,
                             std=IMAGENET_INCEPTION_STD)
                         #tfs.Normalize(mean=[0.5, 0.5, 0.5],
                            #  std=[0.5, 0.5, 0.5]) 
                         ])
    
    if train: return t_tr
    else: return t_val

