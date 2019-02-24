import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim import Adam
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
import time
import os
import copy
import argparse

#Load pre-trained models
vgg16 = models.vgg16(pretrained=True)
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
dropout = 0.2

#define and get arguments
def main(data_dictionary):
    print (‘Enter data dictionary,  %s!’ % data_dictionary)
if __data_dictionary__ == ‘__main__’:
    parser = argparse.ArgumentParser(description = ‘data_dictionary’)
    parser.add_argument (‘data_dictionary’, type=str, help= ‘data dictionary containing data for training and testing, enter it’)
    args = parser.parse_args() 
    main (args.data_dictionary)

def main(save_dir):
    print (‘Enter save directory,  %s!’ % save_dir)
if __save_dir__ == ‘__main__’:
    parser = argparse.ArgumentParser(description = ‘save_dir’)
    parser.add_argument (‘save_dir’, type=str, default=checkpoint_path, help=’directory to save trained model and parameters, enter it’)
    args = parser.parse_args()
    main (args.save_dir)

def main(arch):
    print (‘Enter architecture,  %s!’ % arch)
if __arch__ == ‘__main__’:
    parser = argparse.ArgumentParser(description = ‘arch’)
    parser.add_argument (‘arch’, type=str, default=vgg16, help=’pre-trained models – vgg16, resnet18, alexnet, enter choice’)
    args = parser.parse_args()
    main (args.arch)

def main(learning_rate):
    print (‘Enter learning rate,  %s!’ % learning_rate)
if __learning_rate__ == ‘__main__’:
    parser = argparse.ArgumentParser(description = ‘learning_rate’)
    parser.add_argument (‘learning_rate’, type=float, default=0.01, help=’desired learning rate, enter it’)
    args = parser.parse_args()
    main (args.learning_rate)

def main(epochs):
    print (‘Enter epochs,  %s!’ % epochs)
if __epochs__ == ‘__main__’:
    parser = argparse.ArgumentParser(description = ‘epochs’)
    parser.add_argument (‘epochs’, type=int, default=20, help=’number of epochs, enter it’)
    args = parser.parse_args()
    main (args.epochs)

def main(hidden_layers):
    print (‘Enter hidden_layers,  %s!’ % hidden_layers)
if __hidden_layers__ == ‘__main__’:
    parser = argparse.ArgumentParser(description = ‘hidden_layers’)
    parser.add_argument (‘hidden_layers’, type=list, default=512, help=’desired number of hidden layers, enter it’)
    args = parser.parse_args()
    main (args.hidden_layers)

def main(gpu):
    print (‘Enter true to use GPU,  %s!’ % gpu)
if __gpu__ == ‘__main__’:
    parser = argparse.ArgumentParser(description = ‘gpu’)
    parser.add_argument (‘gpu’, type=bool, default=True, help=’use GPU or CPU to train model, enter true to use GPU or false to use CPU’)
    args = parser.parse_args()
    main (args.gpu)


def main(output):
    print (‘Enter number of outputs,  %s!’ % output)
if __output__ == ‘__main__’:
    parser = argparse.ArgumentParser(description = ‘output’)
    parser.add_argument (‘output’, type=int, default=102, help=’number of outputs, enter it’)
    args = parser.parse_args()
    main (args.output)
  
#define data directories
data_dir = input.data_dir    
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

dataloaders['train'], dataloaders['valid'], dataloaders['test'] = process(train_dir, valid_dir, test_dir)

#load pretrained model and determine input size
model_dict = {"vgg": vgg16, "resnet": resnet18, "alexnet": alexnet}
input_dict = {"vgg": 25088, "resnet": 512, "alexnet": 9218}

model = model_dict[pretrained_model]
input_size = inputsize_dict[pretrained_model]

#freeze parameters
for param in model.parameters():
    param.requires_grad = False
    
    #define transforms for the training, validation, and testing sets
    data_transforms = {
        
        'train': transforms.Compose([            
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
   
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
   
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),}
   
    image_datasets = dict()

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                     for x in ['train', 'valid', 'test']}

    dataloaders = dict()

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) 
                  for x in ['train', 'valid', 'test']}

    dataset_sizes ={x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

    dataset_sizes ={x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes

    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

# Imports and build model

import torchvision.models as models
import argparse
from collections import OrderedDict

model = input.arch

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
for param in model.parameters():
    param.requires_grad = False    
    
    model.classifier = classifier
    model
   
#Training the Model - from Pytorch tutorial (https: //medium.com/@josh_2774/deep-learning-with-pytorch-9574d17ad)
def train_model(model,criteria, optimizer, scheduler,num_epochs=20, device='cuda'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-' *10)
        
        #Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.to(device).train() # Set model to training mode
            else:
                model.to(device).eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))        
        
        # deep copy the model
        if phase == 'valid' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    print()
        
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)
return model.to(device)

# Criteria NLLLoss which is recommended with Softmax final layer
criterion = nn.NLLLoss()

# Observe that all parameters are being optimized
optim = torch.optim.Adam(model.classifier.parameters(), lr=0.01)

# Decay LR by a factor of 0.1 every 4 epochs
sched = lr_scheduler.StepLR(optim, step_size=4, gamma=0.1)

# Number of epochs
eps=20

model_ft = train_model(model, criterion, optim, sched, eps, 'cuda')

# Define calculations
def calc_accuracy(model, data, cuda=False):
    model.eval()
    model.to(device='cuda')    
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[data]):
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # obtain the outputs from the model
            outputs = model.forward(inputs)
            # max provides the (maximum probability, max value)
            _, predicted = outputs.max(dim=1)
            # check the 
            if idx == 0:
                print(predicted) #the predicted class
                print(torch.exp(_)) # the predicted probability
            equals = predicted == labels.data
            if idx == 0:
                print(equals)
                print(equals.float().mean()) 
            
# define check accuracy on test data
print('Beginning accuracy testing...')
t1 = time.time()
correct = 0
total = 0
with torch.no_grad():
    for data in dataloaders['test']:
        print('{}/{}'.format(correct, total))
        images, labels = data
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = ((predicted == labels).float().sum()/batch_size)*100
            
print('Accuracy on test images: %d %%' % (100 * correct / total))
t_total = time.time() - t1;
print('Accuracy test time:  {:.0f}m {:.0f}s'.format(t_total // 60, t_total % 60))

# Save the checkpoint 

model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()

# load checkpoint path and model
def load_model(checkpoint_path):    
    chpt = torch.load(checkpoint_path)
    if chpt['arch'] == 'vgg': 
        model=models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    elif chpt['arch'] == 'resnet':
        model=model.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    elif chpt['arch'] == 'alexnet':
        model=model.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Sorry base architecture not recognized")

#load pretrained model and determine input size
model_dict = {"vgg": vgg16, "resnet": resnet18, "alexnet": alexnet}
input_dict = {"vgg": 25088, "resnet": 512, "alexnet": 9218}

model = model_dict[pretrained_model]
input_size = inputsize_dict[pretrained_model]

torch.save({'arch': input.arch,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            'classifier.pth')
