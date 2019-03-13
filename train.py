#train.py
#final project - part2 - Kim Graffius

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
from PIL import Image

def main():
    input = get_args()
    parser =train_args.get_args()
    kg_args = parser.parse_args()
    
    #define data directories
    data_dir = kg_args.data_dir    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    #load pretrained model and determine input size - code provided from project review
    from torchvision import models
    # then print the model architecture:
    model  = model.densenet201()
    print(model)
    # in the classifier find that there are in_features for each layer. the in_features of the first classifier layer is the value of in_features you should use
    # to access this for densenet201
    print(model.classifier.in_features)

    # Now try this for resnet
    model = models.resnet101()
    print(model)
    # As you can see the classifier of renset is called `fc` to print it's input features 
    print(model.fc.in_features)

    # Now try this for alexnet
    model = models.alexnet()
    print(model)
    # As you can see the first classifier layer is dropout, but we are required first linear layer of classifier and it's in_features
    print(model.classifier[1].in_features)


    model = model_dict[pretrained_model]
        input_size = inputsize_dict[pretrained_model]
    
    #freeze parameters
    for param in model.parameters():
        param.requires_grad = False  
        
    classifier = NeuralNetwork(input_size, output_size, hidden_layers, dropout)
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optim = torch.optim.Adam(model.classifier.parameters(), lr=kg_args.learning_rate)

   
    #load categories for training   
    with open(args.category_names, 'r') as f:
        label_map = json.load(f)
        
    #set output to number of categories 
    output_size = len(cat_to_name)
    print(f"Image classifier has {output_size} categories.")
    

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
          
    #start with CPU
    device = torch.device("cpu")
    #requested GPU
    if kg_args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("GPU is not available - using CPU.")
        nn_model = nn_model.to(device)
        
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
            
            
    #Training the Model 
    train_model(model,criteria, optimizer, scheduler,num_epochs=20, device='cuda'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    droput = kg_args.dropout
    
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
    calc_accuracy(model, data, cuda=False)
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
                   'batch_size': trainloader.batch_size,
                   'output_size': 102,
                   'state_dict': nn_model.state_dict(),
                   'optimizer_dict': optimizer.state_dict(),
                   'classifier':  nn_model.classifier
                   'class_to_idx': nn_model.class_to_idx
                   'arch': kg_args.arch
                   'dropout': kg_args.dropout}

    checkpoint_path = f'{kg_args.save_dir}/{kg_args.save_name}'
    print(f"Saving checkpoint to {checkpoint_path}")

    torch.save(model_state, checkpoint_path)
                                     
   
if __name__ == '__main__':
    
    main()
        
        
        
        
        






