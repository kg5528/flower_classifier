#imports
import argparse

#define and get arguments
def get_args():
    parser = argparse.ArgumentParser(description"Predict Image", usage="python ./predict.py /path/to/image.jpeg checkpoint.pth",  formatter_class=argparse.ArgumentDefaultHelpFormatter)

    parser.add_argument (‘image_path’, action="store", help=’path to image, enter it’)

    parser.add_argument (‘checkpoint’, action="store", help=’path to checkpoint, enter it’)
   
    parser.add_argument (‘--save_dir’, action="store", default=".", dest='save_dir',type=str, help=’directory to save checkpoint file, enter it’)

    parser.add_argument (‘--top_num’, action="store", dest='top_num', type=int, default=3, help=’number topK class to predict, enter it’)
 
    parser.add_argument (‘--category_names’, action="store", dest='category_names', type=str, default=cat_to_name.json, help=’file to convert labels, enter it’)

    parser.add_argument (‘--gpu’, action="store", dest='gpu', type=bool, default=False, help=’use GPU or CPU to train model, enter true to use GPU or false to use CPU’)
    args = parser.parse_args()
    main (args.gpu)
        
    parser.parse_args()
    return parser   

def main():
    print(f'Command line arguments for predict.py. \nTry  "python predict.py -h".')
    if __name__ == '__main__':
        main()

