# predict_args.py 
# final project - part 2 - Kim Graffius


#imports
import argparse

#define and get arguments
def main():
    parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Image Prediction")

    parsser.add_argument('data_dir', action="store", type=str, default='./flowers/', help='path to image, enter it')
    parser.add_argument('checkpoint', action="store", type=str, default='./model_check_point/', help='path to checkpoint, enter it')
    parser.add_argument('save_dir', action="store", type=str, default='.', help='directory to save checkpoint file, enter it')                     
    parser.add_argument('save_name', action="store", type=str, default='.', help='file to save checkpoint, enter it')
    parser.add_argument('category_names', action="store", type=str, default='cat_to_name.json', help='file to convert labels, enter it')
    parser.add_argument('top_num', action="store", type=str, default='3', help='number of topK classes to predict, enter it')
    parser.add_argument('gpu', action="store", type=str, default='False', help='use GPU or CPU to train model, enter True to use GPU or False to use CPU')
 
    return parser.parse_args()

    results = parser.parse_args()
    print('data_dir           = {!r}'.format(results.data_dir))
    print('checkpoint         = {!r}'.format(results.checkpoint))
    print('save_dir           = {!r}'.format(results.save_dir))
    print('save_name          = {!r}'.format(results.save_name))
    print('category_names     = {!r}'.format(results.category_names))
    print('top_num            = {!r}'.format(results.top_num))
    print('gpu                = {!r}'.format(results.gpu))
   
    args = parser.parse_args()
                         
    kg_args = parser.parse_args(args)
    return kg_args
    
    print(parser.parse_args(['data_dir', 'checkpoint', 'save_dir', 'save_name', 'category_names', 'top_num', 'gpu']))
                         

    if __name__ == '__main__':
        main()
