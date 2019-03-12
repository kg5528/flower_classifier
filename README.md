# flower_classifier
AI Programming in Python Nanodegree Final Project Flower Classifier - Kimberly Graffius

Project Summary
For the Final Project for Udacity’s AI Programming with Python Nanodegree program, I built and trained a neural network and created an Image Classifier for 102 different kinds of flowers in two different image classifiers programs (the image classifiers programs are similar but are different file formats).  The second portion of the project includes four files “train.py”, "train_args.py", "predict.py" and “predict_args.py” which allows the user to enter various hyperparameters and 3 different pre-trained networks options into the command line to train and predict a dataset of images. 

Project Part_1 - Development Notebook
In a Jupyter Notebook, a flower dataset of images was downloaded into dataloaders for training, testing, and validation of the neural network. The datasets were transformed in order to increase accuracy, and the images were normalized for the pre-training of the network. The images were resized, cropped, and randomly flipped. Various hyperparameters were defined, such as number of epochs and learning rate, and the model was trained on a VGG16 pretrained network using torchvision.models. Training loss, validation loss, and accuracy were printed and an accuracy of 73.72% was achieved. A checkpoint saved the model, classifier, and hyperparameters. An image of an English Marigold was used to predict the topK 5 possible flower categories and a bar chart was printed that showed exceptional accuracy of the English Marigold classification with the remaining 4 classes showing very low probabilities.

Project Part_2 - Command Line Application
The second portion of the project includes 4 python files: “train.py”, "train_args.py", "predict.py", and “predict_args.py”.  These command line programs allow the user to train a neural network on a dataset of images and then predict the topK classification of a specific image. The "train.py" file uses similar commands that the Development Notebook used in Part_1, but the user has the option to choose VGG16, resnet18, or alexnet for the pre-trained network. Hyperparameters (number of epochs, number of hidden layers, dropout, GPU, number of topK probabilities, etc.) can be changed by the user in the "train_args.py" and "predict_args.py" programs.  The "train.py" program should output training loss, validation loss, and accuracy; and saves the model to a checkpoint. The "predict.py" program uses the checkpoint from the "train.py" program to determine the topK class predictions of the English Marigold image.

