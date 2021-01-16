# -*- coding: utf-8 -*
import os, random
import numpy as np
import matplotlib.pyplot as plt
from PCA import PCA
from dataloader import load_data
from gradientDescent import *
from utils import *

def main():
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
    # (Normailzation is necessary for gradient descent) Or the learnt answer will be biased
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
    print(cost_val_list)
    ad
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

if __name__ == '__main__':
    main()