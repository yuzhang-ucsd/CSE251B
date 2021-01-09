# -*- coding: utf-8 -*
import os, random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PCA import PCA
from dataloader import load_data
from gradientDescent import *

# Convert 2-D image(200x300) to 1-D vector
def flatten_img(imgset):
    new_set = []
    for i in range(len(imgset)):
        new_set.append(imgset[i].flatten())
    return np.array(new_set)

# Split the dataset into (k-2) sets for training, 1 set for validation and 1 set for testing
def split_dataset(dataset,k):
    size = len(dataset)
    cut_train = int((k-2)/k * size) # training cutoff
    cut_val = int(1/k * size) # validation cutoff
    # shuffle the dataset
    idx = np.arange(0,size)
    random.shuffle(idx)
    shuffled_M = dataset[idx]
    # select subsets of data
    train = shuffled_M[:cut_train] # (k-2) training set
    val = shuffled_M[cut_train:(cut_train+cut_val)] # 1 validation set
    test = shuffled_M[(cut_train+cut_val):] # 1 testing set
    return train, val, test

# Check images for top n PCs
def PC_plot(eigen_vectors,num):
    imgs = []
    plt.figure()
    f, ax = plt.subplots(num,1)
    for n in range(num):
        vh = eigen_vectors[n]
        vh_img = np.reshape(vh,(200,300))
        img = Image.fromarray(vh_img)
        imgs.append(img)
        ax[n].imshow(imgs[n])
    plt.show()

# Load data from ./resized/ folder
images, cnt = load_data(data_dir="./resized/")
Minivan = images.get('Minivan')
Convertible = images.get('Convertible')

minivan = flatten_img(Minivan)
convertible = flatten_img(Convertible)


# select train/val/test from class1=Minivan(M)
trainM, valM, testM = split_dataset(minivan,10)
trainM_y = np.ones(len(trainM))
# select train/val/test from class2=Convertible(C)
trainC, valC, testC = split_dataset(convertible,10)
trainC_y = np.zeros(len(trainC))
# combine the datasets
train1 = np.concatenate((trainM, trainC),axis=0) # images
train1_y = np.concatenate((trainM_y, trainC_y), axis=0)
train1_y = np.array([[i] for i in train1_y]) # classes - input y to Logistic regression model
val1 = np.concatenate((valM, valC),axis=0)
test1 = np.concatenate((testM, testC),axis=0)


# Perform PCA on the flattened dataset
# Try using different numbers of components
num_PC = 3
projected, mean_image, top_sqrt_eigen_values, top_eigen_vectors, eigen_vectors = PCA(train1,num_PC)

# Optional: Check images for top n PCs
# PC_plot(eigen_vectors,num_PC)

# Check projections - should be zero mean and unit std
# Answer: this is a good idea because normalization help speed up the gradient descent etc.
# print(np.average(projected,axis=0)) # this is close to zero
# print(np.std(projected,axis=0)) # !=1 ???


# input x to the regression model
train1_x = np.insert(projected, 0, 1, axis=1) # adding x0 = 1 (a ones column)
w = theta(train1_x) # initial weight vector
print(train1_x.shape)
print(train1_y.shape)
print(w.shape)
cost = cost_function(train1_x,train1_y,w)
print(cost)
grad = gradient_one_round(train1_x,train1_y,w)
print(grad)



