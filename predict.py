#predict.py 
#final project  - part2 - Kim Graffius

#imports
import os
import sys
import time
import copy
import json
import predict_args
import torch
from collections import OrderedDict
from torchvision import datasets, transforms, models, utils
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F

#image classification prediction
def main():
    parser = predict_args.get_args()
    kg_args = parser.parse_args()

    #start with CPU
    device = torch.device("cpu")
    #requested GPU
    if kg_args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        
    #category names
    with open(kg_args.cat_names, 'r') as f:
        cat_to_name = json.load(f)
    
    #load model
    checkpoint_model = load_checkpoint(device, kg_args.checkpoint)
    topk_probs, classes = predict(kg_args.image_path, checkpoint_model, kg_args.top_num)
    
    label = top_classes[0]
    prob = top_prob[0]    
    
    #process image
    process_image():
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image
    from PIL import Image
    img = Image.open(kg_args.image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
   # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                     top_margin))
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
       
    return img
    image_path = kg_args.image_path

    predict(image_path, model, top_num=3):
     
    # Unsqueeze to obtain single integer value
    model_input = image_tensor.unsqueeze(0)
    
    # Calculate Probs
    with torch.no_grad():
        output = model.forward(model_input)
    probs = torch.exp(output)
    
    ##probs = torch.exp(model.forward(model_input))
    # get labels from json
    with open('cat_to_name.json', 'r') as f:
        label_map = json.load(f)
        
    # Get topk results
    topk_probs, topk_labs_model = probs.topk(3)
    topk_probs = topk_probs.detach().tolist()[0] 
    topk_labs_model = topk_labs_model.detach().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # Appending topk labels to a list
    topk_labels_org = []
    for i in topk_labs_model:
        topk_labels_org.append(label_map[idx_to_class[i]])
    
    return topk_probs, topk_labels_org
    probs, labels = predict(image_path, model) 

if __name__ == '__main__':
    main()


