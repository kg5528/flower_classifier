!/usr/bin/env python3

#final project - part2
__author__ = "Kim Graffius <kg5528@att.com>"
__version__ = "1.0"

#imports
import os
import sys
import time
import copy
import json
import train_args
import torch
from collections import OrderedDict
from torchvision import datasets, transforms, models, utils
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
#dropout = 0.2



def main():
    parser = train_args.get_args()
    parser.add_argument('version', action='version', version='%(prog)s  ' + __version__ + __author__)
    kg_args = parser.parse_args()
    
    #check for args
    
    #check for data directory
    if not os.path.isdir(kg_args.data_dir):
        print(f'Directory {kg_args.data_dir} was not found.')
        exit(1)
        
    #check for save directory  and create if needed 
    if not os.path.isdir(kg_args.save_dir):
        print(f'Directory {kg_args.save_dir} does not exist. Creating now....')
        os.makedirs(kg_args.save_dir)
        
    #load categories for training   
    with open(kg_args.category_names, 'r') as f:
       cat_to_name = json.load(f)
        
    #set output to number of categories 
    output_size = len(cat_to_name)
    print(f"Image classifier has {output_size} categories.")
    
    #define data directories
    data_dir = kg_args.data_dir    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    dataloaders['train'], dataloaders['valid'], dataloaders['test'] = process(train_dir, valid_dir, test_dir)
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

        class_names = image_datasets['train'].classes    
          
        #determine model details
        if not kg_args.arch.startswith("vgg") and not kg_args.arch.startswith("resnet") and not kg_args.arch.startswith("alexnet"):
            print("Only supporting VGG, Resnet, and Alexnet")
            exit(1)
        print(f"Using a pre-trained {kg_args.arch} model.")
        nn.model = models.__dict__[kg_args.arch](pretrained=True)    

        #pre-trained models supported
        vgg16 = models.vgg16(pretrained=True)
        resnet18 = models.resnet18(pretrained=True)
        alexnet = models.alexnet(pretrained=True)

        #load pretrained model and determine input size
        model_dict = {"vgg": vgg16, "resnet": resnet18, "alexnet": alexnet}
        input_dict = {"vgg": 25088, "resnet": 512, "alexnet": 9218}

        #model = model_dict[pretrained_model]
        input_size = inputsize_dict[pretrained_model]

    #freeze parameters
    for param in model.parameters():
        param.requires_grad = False
 
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(4096, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    for param in model.parameters():
        param.requires_grad = False    
    
        model.classifier = classifier
        
        #set gradients of all parameters to zero
        nn_model.zero_grad

        # Criteria NLLLoss which is recommended with Softmax final layer
        criterion = nn.NLLLoss()

    # Observe that all parameters are being optimized
    print(f"Setting optimizer learning rate to {kg_args.learning_rate}.")
    optim = torch.optim.Adam(model.classifier.parameters(), lr=kg_args.learning_rate)

    #start with CPU
    device = torch.device("cpu")
    #requested GPU
    if kg_args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("GPU is not available - using CPU.")
        nn_model = nn_model.to(device)
        
   
    #Training the Model - from Pytorch tutorial (https: //medium.com/@josh_2774/deep-learning-with-pytorch-9574d17ad)
def train_model(model,criteria, optimizer, scheduler,num_epochs=20, device='cuda'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(kg_args.num_epochs):
        print('Epoch {}/{}'.format(epoch, kg_args.num_epoch - 1))
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

    # Decay LR by a factor of 0.1 every 4 epochs
    sched = lr_scheduler.StepLR(optim, step_size=4, gamma=0.1)

    # Number of epochs
    eps=kg_args.num_epochs

    model_ft = train_model(model, criterion, optim, sched, eps, 'cuda')

    # Define calculations
def calc_accuracy(model, data, cuda=False)
    model.eval()
    model.to(device='cuda')    
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[data]):
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # obtain the outputs from the model
            outputs = model.forward(inputs)
            # max provides the (maximum probability, max value)
            _, predicted = outputs.max(dim=1
            # check 
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

    nn_model.class_to_idx = image_datasets['train'].class_to_idx
    model_state = {'epoch': kg_args.num_epochs,
                   'state_dict': nn_model.state_dict(),
                   'optimizer_dict': optimizer.state_dict(),
                   'classifier':  nn_model.classifier
                   'class_to_idx': nn_model.class_to_idx
                   'arch': kg_args.arch}

    save_location = f'{kg_args.save_dir}/{kg_args.save_name}.pth'
    print(f"Saving checkpoint to {save_location}")

    torch.save(model_state, save_location)

if __name__ == '__main__':
    
    main()
