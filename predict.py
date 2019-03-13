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
    
    label = top_classes[0]
    prob = top_prob[0]   
    
def load_model(checkpoint_path):
    chpt = torch.load(args.checkpoint_path)
    
    if chpt['args.arch'] == 'densenet201':
        model = models.densenet201(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
    elseif chpt['args.arch'] == 'resnet101':
        model = models.'resnet101'(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
     
    elseif chpt['args.arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
    else:
        print("Sorry base architecture not recognized")        
    
    model.class_to_idx = chpt['class_to_idx']
    
    classifier = NeuralNetwork(input_size, output_size, hidden_layers, dropout)
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optim = torch.optim.Adam(model.classifier.parameters(), lr=kg_args.learning_rate)
    
    #code for NeuralNetwork classifier
    class NeuralNetwork(nn.Module):
        #define layers of neural network
        def __init__(self, input_size, output_size, hidden_layers, dropout):
            super().__init__()
            #input size to hidden layer
            self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
            #add hidden layers
            i = 0
            j = len(hidden_layers)-1
            
            while i != j:
                l = [hidden_layers[i], hidden_layers[i+1]]
                self.hidden_layers.append(nn.Linear(l[0], l[1]))
                i+=1
            #check to make sure hidden layers are formatted correctly
            for each in hidden_layers:
                print(each)
            #last hidden layer to output
            self.output = nn.Linear(hidden_layers[j], output_size)
            self.dropout = nn.Dropout(p = dropout)
        #feed forward method
        def forward(self, tensor):
            for linear in self.hidden_layers:
                tensor = F.relu(linear(tensor))
                tensor = self.dropout(tensor)
            tensor - self.output(tensor)
            #log softmax
            return F.log_softmax(tensor, dim=1)    
        
    model.load_state_dict(chpt['state_dict'])
        
    return model
                    
model = load_model('classifier.pth')
calc_accuracy(model, 'test', True)
    
    #process image
def process_image():
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
    

def predict(image_path, model, top_num=3):
    
    image_path = kg_args.image_path
     
    # Unsqueeze to obtain single integer value
    model_input = image_tensor.unsqueeze(0)
    
    # Calculate Probs
    with torch.no_grad():
        output = model.forward(model_input)
    probs = torch.exp(output)
    
    ##probs = torch.exp(model.forward(model_input))
    # get labels from json
    with open(args.category_names, 'r') as f:
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
    

if __name__ == '__main__':
    main()
