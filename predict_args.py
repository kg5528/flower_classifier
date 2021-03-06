# predict_args.py 
# final project - part 2 - Kim Graffius


#imports
import argparse

#define and get arguments
def main():
    parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Image Prediction")

    parsser.add_argument('image_path', action="store", type=str, default='./flowers/train/5/image_05151.jpg', help='path to image, enter it')
    parser.add_argument('checkpoint_path', action="store", type=str, default='./model_check_point/', help='path to checkpoint, enter it')
    parser.add_argument('category_names', action="store", type=str, default='cat_to_name.json', help='file to convert labels, enter it')
    parser.add_argument('top_num', action="store", type=str, default='3', help='number of topK classes to predict, enter it')
    parser.add_argument('gpu', action="store", type=str, default='False', help='use GPU or CPU to train model, enter True to use GPU or False to use CPU')
 
    return parser.parse_args()

    results = parser.parse_args()
    print('image_path         = {!r}'.format(results.data_dir))
    print('checkpoint_path    = {!r}'.format(results.checkpoint))
    print('category_names     = {!r}'.format(results.category_names))
    print('top_num            = {!r}'.format(results.top_num))
    print('gpu                = {!r}'.format(results.gpu))
   
    args = parser.parse_args()
                         
    kg_args = parser.parse_args(args)
    return kg_args
    
    print(parser.parse_args(['image_path', 'checkpoint_path', 'category_names', 'top_num', 'gpu']))
                         

    if __name__ == '__main__':
        main()
