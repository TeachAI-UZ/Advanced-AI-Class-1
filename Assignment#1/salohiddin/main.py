import torch, argparse, yaml, timm
from transforms import get_transforms
from dataset import get_dl
from train import train
from lion_pytorch import Lion
from inference import inference
from utils import create_model_dir


def run(args):
    
    # Get train arguments    
    root = args.root
    batch_size = args.batch_size
    device = args.device
    lr = args.learning_rate
    model_name = args.model_name
    epochs = args.epochs
    log_dir = args.log_dir
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}")
    
    
    # Get train and validation transformations 
    train_transformations, valid_transformations= get_transforms(train=True), get_transforms(train=False)
    
    # Get class names, number of classes, train and validation dataloaders
    cls_names, num_classes, tr_dl, val_dl, test_dl = get_dl(root, batch_size, valid_transformations)
    # _,         _,           tr_dl,      _,        _ = get_dl(root, batch_size, train_transformations)
    
    # Apply training transformations to ONLY TRAIN DATALOADER
    tr_dl.transform = train_transformations


    print(f"Number of classes in the dataset: {num_classes}\n")
    
    # Initialize model, loss_function, and optimizer    
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  #torch.optim.Adam or Lion
    

    # Create the model DIRECTORY if not available
    log_dir = create_model_dir(log_dir)
    
    # Train model
    train(model, tr_dl, val_dl, criterion, optimizer, epochs, batch_size, lr, device, 
          log_dir)   
    
    # Test the model on unseen data
    inference(model_name, num_classes, log_dir, device, test_dl, cls_names)
   
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Image Classification Training Arguments')
    parser.add_argument("-r", "--root", type = str, default = 'simple_classification', help = "Path to the data")
    parser.add_argument("-bs", "--batch_size", type = int, default = 64, help = "Mini-batch size")
    parser.add_argument("-mn", "--model_name", type = str, default = 'rexnet_150', help = "Model name for backbone")
    parser.add_argument("-d", "--device", type = str, default = 'cuda:3', help = "GPU device number")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-3, help = "Learning rate value")
    parser.add_argument("-ld", "--log_dir", type = str, default = 'resnet50', help = "Directory to save the model outputs" )
    parser.add_argument("-eps", "--epochs", type = int, default = 50, help = "Epoch count")
    args = parser.parse_args() 
    
    run(args) 


# How to run the code and train and test any model you specify? >>>
# Copy and paste the following in the terminal without "#"

 # python main.py  --root="/mnt/C6E2920FE29203B9/din/Deep_l/1_classification/1_data/nuts/train/"  -bs=192  -mn='resnet50' -d='cuda:0' -ld="resnet50_dir" -eps=2