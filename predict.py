import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim import Adam
import pandas as pd
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import torchvision
from torchvision import datasets, transforms, models, utils
import torchvision.models as models
from PIL import Image
import json
from collections import OrderedDict
import seaborn as sns
import time
import os
import copy
import argparse

   #define get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="path to image")
    parser.add_argument("checkpoint", type=str, help="checkpoint for trained model")
    parser.add_argument("--top_num", type=int, default=3, help="topk classes to predict")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="file to convert labels")
    parser.add_argument("--gpu", type=bool, default=True, help="use GPU (true) or CPU (false)")
    
    return parser.parse_args()

def main():
    #get arguments
    input = get_args()
    path_to_image = input.image_path
    checkpoint = input.checkpoint
    top_num = input.top_num
    cat_names = input.category_names
    gpu = input.gpu
    
    return parser.parse_args()
        
    #category names
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
        model = load(checkpoint)
        img = Image.open(path_to_image)
        image = process_image(img)
        topk_probs, classes = predict(path_to_image, model, top_num)
        check(image, path_to_image, model)
        
def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image
    from PIL import Image
    img = Image.open(path_to_image)
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

    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img

image_path = input.image_path

def predict(image_path, model, top_num=3):
    # Process image
    img = process_image(image_path)
    
    # Check if cuda is available
    cuda = torch.cuda.is_available()
    
    if cuda:
        #move model to GPU
        model.cuda()
            
        # Numpy -> Tensor GPU
        image_tensor = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        print("Using GPU to predict")
        
    else:
        #move model to CPU
        model.cpu()
        
        # Numpy -> Tensor CPU
        image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        print("Using CPU to predict")
        
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

def plot_solution(image_path, model):
    
    # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    
    # Set up title    
    flower_num = image_path.split('/')[2]
    with open('cat_to_name.json', 'r') as f:
        label_map = json.load(f)
    title_mapped = str.title(label_map[flower_num])
    
    # Plot flower
    ax.set_title(title_mapped)
    img = process_image(image_path)
    imshow(img, ax)

    # Make prediction
    probs, labels = predict(image_path, model) 
    
    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y = [str.title(label) for label in labels], color="lightblue");
    plt.show()
    
    image_path = input.image_path
    plot_solution(image_path, model)
