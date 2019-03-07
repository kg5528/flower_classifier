import argparse

def main(name):
    
    if __name__ == '__main__':
    
        parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Train and Save Image Classifier")
    
        parser.add_argument('data_dir', action="store", type=str, default='./flowers/', help='directory containing data for training, enter it')
        parser.add_argument('save_dir', action="store", type=str, default='./model_check_points/', help='directory to save trained model, enter it')
        parser.add_argument('save_name', action="store", type=str, default='checkpoint.pth', help='checkpoint filename, enter it')
        parser.add_argument('category_names', action="store", type=str, default='cat_to_name.json', help='file to convert labels, enter it')
        parser.add_argument('arch', action="store", type=str, default='vgg16', help='pre-trained models - vgg16, resnet18, or alexnet, enter choice')
        parser.add_argument('gpu', action="store", type=str, default='False', help='use GPU or CPU to train model, enter True to use GPU or False to use CPU')
        parser.add_argument('learning_rate', action="store", type=float, default='0.01', help='desired learning rate, enter it')
        parser.add_argument('num_epochs', action="store", type=int, default='20', help='number of epochs, enter it')   
        parser.add_argument('hidden_layers', action="store", type=int, default='512', nargs='+', help='desired number of hidden layers, enter it')
        parser.add_argument('dropout', action="store", type=float, default='0.2', help='drop out rate, enter it')
    
        args = parser.parse_args(['data_dir', 'save_dir', 'save_name', 'category_names', 'arch', 'gpu', 'learning_rate', 'num_epochs', 'hidden_layers', 'dropout'])
        print(parser.parse_args(['data_dir', 'save_dir', 'save_name', 'category_names', 'arch', 'gpu', 'learning_rate', 'num_epochs', 'hidden_layers', 'dropout']))
    
if __name__ == '__main__':
    main('name')
