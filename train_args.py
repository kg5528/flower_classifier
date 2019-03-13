# train_args.py 
# final project - part 2 - Kim Graffius

import argparse

def main():
    
    parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Train and Save Image Classifier")
    parser.add_argument('data_dir', action="store", type=str, default='/home/workspace/aipnd-project/flowers/', help='directory containing data for training, enter it')
    parser.add_argument('save_dir', action="store", type=str, default='/home/workspace/aipnd-project/model_check_points/', help='directory to save trained model, enter it')
    parser.add_argument('save_name', action="store", type=str, default='checkpoint.pth', help='checkpoint filename, enter it')
    parser.add_argument('category_names', action="store", type=str, default='cat_to_name.json', help='file to convert labels, enter it')
    parser.add_argument('arch', action="store", type=str, default='densenet201', help='pre-trained models densenet201, resnet101, or alexnet, enter choice')
    parser.add_argument('gpu', action="store", type=str, default='False', help='use GPU or CPU to train model, enter True to use GPU or False to use CPU')
    parser.add_argument('learning_rate', action="store", type=str, default='0.01', help='desired learning rate, enter it')
    parser.add_argument('num_epochs', action="store", type=str, default='20', help='number of epochs, enter it')   
    parser.add_argument('hidden_layers', action="store", type=str, default='512', nargs='+', help='desired number of hidden layers, enter it')
    parser.add_argument('dropout', action="store", type=str, default='0.2', help='drop out rate, enter it')
    
    return parser.parse_args()
    
    results = parser.parse_args()
    print('data_dir           = {!r}'.format(results.data_dir))
    print('save_dir           = {!r}'.format(results.save_dir))
    print('save_name          = {!r}'.format(results.save_name))
    print('category_names     = {!r}'.format(results.category_names))
    print('arch               = {!r}'.format(results.arch))
    print('gpu                = {!r}'.format(results.gpu))
    print('learning_rate      = {!r}'.format(results.learning_rate))
    print('num_epochs         = {!r}'.format(results.num_epochs))
    print('hidden_layers      = {!r}'.format(results.hidden_layers))
    print('dropout            = {!r}'.format(results.dropout))
    
    kg_args = parser.parse_args(args)
    print(parser.parse_args(['data_dir', 'save_dir', 'save_name', 'category_names', 'arch', 'gpu', 'learning_rate', 'num_epochs', 'hidden_layers', 'dropout']))
    
if __name__ == '__main__':
    main()
