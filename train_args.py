!/usr/bin/env python3

#imports
import argparse

#support architectures
supported_arch = ['vgg16', 'resnet18', 'alexnet']

#define and get arguments
def get_args():
    parser = argparse.ArgumentParser(description"Train and Save Image Classifier", usage="python ./train.py ./flowers/train --gpu --learning_rate 0.01 --hidden_units 512 --num_epochs 20", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument (‘data_dir’, action="store", dest="data_dir", type=str, help=‘data directory containing data for training and testing, enter it’)
 
    parser.add_argument (‘save_dir’, action="store", default=".", dest="save_dir",type=str, help=’directory to save trained model and parameters, enter it’)
   
    parser.add_argument (‘save_name’, action="store", default=".", dest="save_name",type=str, default=checkpoint, help=’checkpoint filename, enter it’)

    parser.add_argument (‘category_names’, action="store", dest="category_names", type=str, default="cat_to_name.json", help=’file to convert labels, enter it’)

    parser.add_argument (‘arch’, action="store", dest="arch", type=str, default=vgg16, help=’pre-trained models – vgg16, resnet18, alexnet, enter choice’)

    parser.add_argument (‘gpu’, action="store", type=bool, dest="gpu", default=False, help=’use GPU or CPU to train model, enter true to use GPU or false to use CPU’)
    
    hyper = parser.add_argument_group('hyper')

    hyper.add_argument (‘learning_rate’, action="store", dest="learning_rate", type=float, default=0.01, help=’desired learning rate, enter it’)

    hyper.add_argument (‘num_epochs’, action="store", dest="num_epochs", type=int, default=20, help=’number of epochs, enter it’)

    hyper.add_argument (‘hidden_layers’, action="store", dest="hidden_layers", type=int, default=512, nargs='+', help=’desired number of hidden layers, enter it’)
  
    parser.parse_args()
    return parser

def main():
    print(f'Command line arguments for train.py. \nTry  "python train.py -h".')
    if __name__ == '__main__':
        main()
