This repository contains Python code for the thesis project 'Optimisation of a Convolutional Neural' submitted in partial fulfilment of the requirements of the award of Master of Engineering in Electrical and Electronic Engineering at the University of Aberdeen
This repository is meant to display the code used to build the optimisation strategy and train the models. However, due to the size of the training data, grasping dataset is not included. Therefore, the system cannot be tested (without the grasping dataset).
File hierarchy:
utills - Functions to ease use of the grasping dataset.
cleanup_data - Method to clean up the dataset from corrupt data and screws.
hough_transform - Method to process images to estimate grasp geometrically. 
modeltesting - Functions to test the model, find discriminative threshold, compute ROC and confusion matrix.
modeltraining - Functions to collect the data, define and compile CNN model. Train the CNN model.
nnmodel - CNN architecture without integrated Bayesian Optimisation.
nnmodel_keras - CNN architecture with integrated Bayesian Optimisation.
