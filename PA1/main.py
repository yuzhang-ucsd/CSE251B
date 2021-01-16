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

# Check images for top 4 PCs
def PC4_plot(eigen_vectors):
    imgs = []
    for n in range(4):
        vh = eigen_vectors[n]
        vh_img = np.reshape(vh,(200,300))
        img = Image.fromarray(vh_img)
        imgs.append(img)
    plt.figure()
    fig, ax = plt.subplots(2,2)
    fig.suptitle('Figure2: Images of top 4 principal components')
    ax[0, 0].imshow(imgs[0])
    ax[0, 0].set_title('Principal Component 0')
    ax[0, 1].imshow(imgs[1])
    ax[0, 1].set_title('Principal Component 1')
    ax[1, 0].imshow(imgs[2])
    ax[1, 0].set_title('Principal Component 2')
    ax[1, 1].imshow(imgs[3])
    ax[1, 1].set_title('Principal Component 3')
    # plt.show()
    plt.savefig('./figures/Q5b_top4PCs.png')

# Generate y vectors (class=0/1) for train/val/test set
def generate_y(dataM,dataC):
    dataM_y = np.ones(len(dataM))
    dataC_y = np.zeros(len(dataC))
    data_y = np.concatenate((dataM_y, dataC_y), axis=0)
    data_y = np.array([[i] for i in data_y])  # classes - input y to Logistic regression model
    return data_y

# Apply PCs computed on the training set to val/testing set to extract small number of representative features
def apply_PCA(input_dataset,mean_image,top_eigen_vectors,top_sqrt_eigen_values):
    msd = input_dataset - mean_image # M x d
    projected = np.matmul(msd, top_eigen_vectors)/top_sqrt_eigen_values
    PACed_x = np.insert(projected, 0, 1, axis=1)
    return PACed_x



# Load data from ./resized/ folder
images, cnt = load_data(data_dir="./resized/")
Minivan = images.get('Minivan')
Convertible = images.get('Convertible')

minivan = flatten_img(Minivan)
convertible = flatten_img(Convertible)


# select train/val/test from class1=Minivan(M)
trainM, valM, testM = split_dataset(minivan,10)
# select train/val/test from class2=Convertible(C)
trainC, valC, testC = split_dataset(convertible,10)
# combine the datasets
train1 = np.concatenate((trainM, trainC),axis=0) # images
val1 = np.concatenate((valM, valC),axis=0)
test1 = np.concatenate((testM, testC),axis=0)
# generate input y from the datasets
train1_y = generate_y(trainM,trainC)
val1_y = generate_y(valM,valC)
test1_y = generate_y(testM,testC)

# Perform PCA on the flattened training set
# Try using different numbers of components
num_PC = 10
projected, mean_image, top_sqrt_eigen_values, top_eigen_vectors, eigen_vectors = PCA(train1,num_PC) # PCA performed only on the training set

# Check images for top 4 PCs
PC4_plot(eigen_vectors)


# Check projections - should be zero mean and unit std
# Answer: this is a good idea because normalization help speed up the gradient descent etc.
# print(np.average(projected,axis=0)) # this is close to zero
# print(np.std(projected,axis=0)) # !=1 ???


# generate input x from the datasets
train1_x = np.insert(projected, 0, 1, axis=1) # adding x0 = 1 (a ones column)
val1_x = apply_PCA(val1,mean_image,top_eigen_vectors,top_sqrt_eigen_values)
test1_x = apply_PCA(test1,mean_image,top_eigen_vectors,top_sqrt_eigen_values)

# print(train1_x.shape)
# print(train1_y.shape)
# print(w.shape)

# initial weight vector
w = theta(train1_x)
# define learning rate and total number of epochs
lr = 5
M = 600
# lists that record cost
cost_train_list = []
cost_val_list = []
cost_test_list = []
# lists that record accuracy
acc_train_list = []
acc_val_list = []
acc_test_list = []

# loop M epochs
for epoch in range(M):
    # tran the model with training set (train_x and train_y)
    grad, w = stepwise_gradient(train1_x,train1_y,w,lr)
    # compute and record cost and accuracy
    # training set
    cost_train = cost_function(train1_x, train1_y, w)
    cost_train_list.append(cost_train)
    acc_train = predict_acc(train1_x, train1_y, w)
    acc_train_list.append(acc_train)
    # acc_train =
    # validation set
    cost_val = cost_function(val1_x,val1_y,w) # test performance on val every epoch
    cost_val_list.append(cost_val)
    acc_val = predict_acc(val1_x, val1_y, w)
    acc_val_list.append(acc_val)
    # testing set
    cost_test = cost_function(test1_x, test1_y, w)
    cost_test_list.append(cost_test)
    acc_test = predict_acc(test1_x, test1_y, w)
    acc_test_list.append(acc_test)
# save the best model
val, idx = min((val, idx) for (idx, val) in enumerate(cost_val_list))

print(cost_test_list[idx],acc_test_list[idx])

# Plot: Cost against Epochs
plt.figure()
plt.plot(cost_train_list, 'b', label='training error')
plt.plot(cost_val_list, 'r', label='validation error')
plt.xlabel('M epochs')
plt.ylabel('Cost')
plt.title('Training and validation loss in one run')
plt.legend()
# plt.show()
plt.savefig('./figures/Q5b_curves.png')