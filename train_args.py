import argparse
from image_classifier_project import *

#support architectures
supported_arch = ['vgg16', 'resnet18', 'alexnet']


def readable_dir(data_dir):
    if not os.path.isdir(data_dir):
        raise Exception("readable_dir:{0} is not a valid path".format(data_dir))
    if os.access(data_dir, os.R_OK):
        return data_dir
    else:
        raise Exception("readable_dir:{0} is not a readable dir".format(data_dir))

def readable_dir(save_dir):
    if not os.path.isdir(save_dir):
        raise Exception("readable_dir:{0} is not a valid path".format(save_dir))
    if os.access(save_dir, os.R_OK):
        return save_dir
    else:
        raise Exception("readable_dir:{0} is not a readable dir".format(save_dir))
   
def readable_file(save_name):
    if not os.path.isdir(save_name):
        raise Exception("readable_file:{0} is not a valid path".format(save_name))
    if os.access(save_name, os.R_OK):
        return save_name
    else:
        raise Exception("readable_file:{0} is not a readable file".format(save_name))
        
def readable_file(category_name):
    if not os.path.isdir(category_names):
        raise Exception("readable_file:{0} is not a valid path".format(category_names))
    if os.access(category_names, os.R_OK):
        return category_names
    else:
        raise Exception("readable_file:{0} is not a readable file".format(category_names))       

    # Set the default hyperparameters if none given
    if in_arg.hidden_units==[]:
        in_arg.hidden_units=[4096, 4096];
    
        
    save_location = f'{kg_args.save_dir}/{kg_args.save_name}.pth'    
    kg_args = parser.parse_args()
    
     # Get the transfer model (vgg11, vgg13, vgg16, vgg19_bn and alexnet)
    model = get_model(hyperparameters['architecture'])

    # Create the dataloaders for training and testing images
    # Also, create the classifier based on the given inputs and attach it to the transfer model
    model, train_dataloader, test_dataloader = model_config(kg.data_dir, 
                                                            model,  
                                                            hyperparameters['hidden_layers'],
                                                            hyperparameters['dropout'])

    if kg_args.gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    # Print the relevant parameters to the output window before training
    print("\n==============================================================\n")
    print("Training data:                {}".format(kg_args.data_dir + '/train'))
    print("Validation data:              {}".format(kg_args.data_dir + '/valid'))
    print("Testing data:              {}".format(kg_args.data_dir + '/test'))
    print("Checkpoint will be saved to:  {}".format(save_name))
    print("Device:                       {}".format(device))

    print("Transfer model:               {}".format(hyperparameters['arch']))
    print("Hidden layers:                {}".format(hyperparameters['hidden_layers']))
    print("Learning rate:                {}".format(hyperparameters['learning_rate']))
    print("Dropout probability:          {}".format(hyperparameters['dropout']))
    print("Epochs:                       {}".format(hyperparameters['num_epochs']))
print("\n==============================================================\n")
    
    
#get arguments
kg_args = parse_input_args()
    
    
    
# Set the inputs to the hyperparameters
#users to set hyperparameters for learning rate, number of hidden units, and training epochs
hyperparameters = {'arch': kg_args.arch,
                   'gpu': kg_args.gpu,
                   'learning_rate': kg_args.learning_rate,
                   'hidden_layers': kg_args.hidden_units,
                   'num_epochs': kg_args.num_epochs,
                   'arch': kg_args.arch,
                   'dropout': kg_args.dropout}        
          
#define and get arguments
def get_args():
    parser = argparse.ArgumentParser(description='Train and Save Image Classifier', 'data_dir', 'save_dir', 'save_name', 'category_names', 'arch', 'gpu', 'learning_rate', 'num_epochs', 'hidden_units', 'dropout', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    
    parser.add_argument('-l', '--launch_directory', type=readable_dir, default='./flowers/')
    
    parser.add_argument('-l', '--launch_directory', type=readable_dir, default='./model_check_points/')
    
    parser.add_argument('-l', '--launch_file', type=readable_file, default='checkpoint.pth')
                       
    parser.add_argument('-l', '--launch_file', type=readable_file, default='cat_to_name.json')
       
    hyper = parser.add_argument_group('hyper')
   
    hyper.add_argument(‘arch’, action="store", dest="arch", type=str, default=vgg16, help=’pre-trained models – vgg16, resnet18, alexnet, enter choice’)

    hyper.add_argument(‘gpu’, action="store", type=bool, dest="gpu", default=False, help=’use GPU or CPU to train model, enter true to use GPU or false to use CPU’)
    
    hyper.add_argument(‘learning_rate’, action="store", dest="learning_rate", type=float, default=0.01, help=’desired learning rate, enter it’)

    hyper.add_argument(‘num_epochs’, action="store", dest="num_epochs", type=int, default=20, help=’number of epochs, enter it’)

    hyper.add_argument(‘hidden_layers’, action="store", dest="hidden_layers", type=int, default=512, nargs='+', help=’desired number of hidden layers, enter it’)
    
    hyper.add_argument('dropout', action="store", dest"dropout", type=float, default=0.2, help='drop out rate, enter it')
  
    parser.parse_args()
    return parser

    #kg_args = parser.parse_args()
    #return kg_args

def main():
    print(f'Command line arguments for train.py. \nTry  "python train.py -h".')
    if __name__ == '__main__':
        main()

  
