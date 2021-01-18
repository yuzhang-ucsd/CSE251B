from gradientDescent import *
from utils import *
import numpy as np
from PCA import PCA

#The training functions for softmax regression
def Softmax(k, minivan, convertible, pickup, sedan, lr, M, num_PC):
    trainM, valM, testM = split_dataset_k_fold(minivan, k)
    trainC, valC, testC = split_dataset_k_fold(convertible, k)
    trainP, valP, testP = split_dataset_k_fold(pickup, k)
    trainS, valS, testS = split_dataset_k_fold(sedan, k)
    cost_train = np.zeros([k, M])
    acc_train = np.zeros([k, M])
    cost_val = np.zeros([k, M])
    acc_val = np.zeros([k, M])
    cost_test = np.zeros([k, M])
    acc_test = np.zeros([k, M])
    final_acc = np.zeros([k,1])
    Confusion_Matrix = np.zeros([4,4])
    for i in range(k):
        print('This is %dth iteration.' %(i))
        cost_train_list, acc_train_list, cost_val_list, acc_val_list, cost_test_list, acc_test_list, final_acc_value, confu_matrix = Softmax_kernel(i, lr, M, num_PC, trainM[i], valM[i], testM[i], trainC[i], valC[i], testC[i], trainP[i], valP[i], testP[i], trainS[i], valS[i], testS[i])
        Confusion_Matrix += confu_matrix
        cost_train[i,:] = np.array(cost_train_list)
        acc_train[i,:] = np.array(acc_train_list)
        cost_val[i,:] = np.array(cost_val_list)
        acc_val[i,:] = np.array(acc_val_list)
        cost_test[i,:] = np.array(cost_test_list)
        acc_test[i,:] = np.array(acc_test_list)
        final_acc[i] = final_acc_value
    for i in range(4):
        Confusion_Matrix[i,:] /= np.sum(Confusion_Matrix, axis = 1)[i]
    return cost_train, acc_train, cost_val, acc_val, cost_test, acc_test, np.mean(final_acc), Confusion_Matrix

def Softmax_kernel(i, lr, M, num_PC, trainM, valM, testM, trainC, valC, testC, trainP, valP, testP, trainS, valS, testS):
    # combine the datasets
    train1 = np.concatenate((trainM, trainC, trainP, trainS),axis = 0) # images
    val1 = np.concatenate((valM, valC, valP, valS),axis = 0)
    test1 = np.concatenate((testM, testC, testP, testS),axis = 0)
    # generate input y from the datasets
    train1_y = generate_y_softmax(trainM, trainC, trainP, trainS)
    val1_y = generate_y_softmax(valM, valC, valP, valS)
    test1_y = generate_y_softmax(testM, testC, testP, testS)

    # Perform PCA on the flattened training set
    # Try using different numbers of components
    projected, mean_image, top_sqrt_eigen_values, top_eigen_vectors, eigen_vectors = PCA(train1,num_PC) # PCA performed only on the training set
    # PC4_plot_softmax(eigen_vectors, i)
    train1_x = np.insert(projected, 0, 1, axis = 1) # adding x0 = 1 (a ones column)
    val1_x = apply_PCA(val1,mean_image,top_eigen_vectors,top_sqrt_eigen_values)
    test1_x = apply_PCA(test1,mean_image,top_eigen_vectors,top_sqrt_eigen_values)

    # initial weight vector
    w = theta_softmax(train1_x, 4)
    # lists that record cost
    cost_train_list = []
    cost_val_list = []
    cost_test_list = []
    # lists that record accuracy
    acc_train_list = []
    acc_val_list = []
    acc_test_list = []
    w_list = []
    # loop M epochs
    for epoch in range(M):
        # tran the model with training set (train_x and train_y)
        w = stepwise_gradient_softmax(train1_x, train1_y, w, lr)
        w_list.append(w)
        # compute and record cost and accuracy
        # training set
        cost_train = cost_function_softmax(train1_x, train1_y, w)
        cost_train_list.append(cost_train)
        acc_train = predict_acc_softmax(train1_x, train1_y, w)
        acc_train_list.append(acc_train)
        # # validation set
        cost_val = cost_function_softmax(val1_x,val1_y,w) # test performance on val every epoch
        cost_val_list.append(cost_val)
        acc_val = predict_acc_softmax(val1_x, val1_y, w)
        acc_val_list.append(acc_val)
        # testing set
        cost_test = cost_function_softmax(test1_x, test1_y, w)
        cost_test_list.append(cost_test)
        acc_test = predict_acc_softmax(test1_x, test1_y, w)
        acc_test_list.append(acc_test)
    # save the best model
    val, idx = min((val, idx) for (idx, val) in enumerate(cost_val_list))  #Access the index associated with the lowest loss on the ValidSet
    # generate the confusion matrix based on the best parameters
    w = w_list[idx]
    ConfusionMatrix = GeneConMatrix(test1_x, test1_y, w)
    # visualize the weights
    Car_Image_plot(top_eigen_vectors, w[1:,:], i)
    return cost_train_list, acc_train_list, cost_val_list, acc_val_list, cost_test_list, acc_test_list, acc_test_list[idx], ConfusionMatrix

#The training functions for Logistic regression
#Main training procejures for CrossRun
def CrossRun(k, minivan, convertible, lr, M, num_PC):
    # define learning rate and total number of epochs
    trainM, valM, testM = split_dataset_k_fold(minivan, k)
    trainC, valC, testC = split_dataset_k_fold(convertible, k)
    cost_train = np.zeros([k, M])
    acc_train = np.zeros([k, M])
    cost_val = np.zeros([k, M])
    acc_val = np.zeros([k, M])
    cost_test = np.zeros([k, M])
    acc_test = np.zeros([k, M])
    final_acc = np.zeros([k,1])
    for i in range(k):
        print('This is %dth iteration.' %(i))
        #the first parameter corresponds to the index of the output image
        cost_train_list, acc_train_list, cost_val_list, acc_val_list, cost_test_list, acc_test_list, final_acc_value = OneRunKernel(i, lr, M, num_PC, trainM[i], valM[i], testM[i], trainC[i], valC[i], testC[i], CrossValid = True)
        cost_train[i,:] = np.array(cost_train_list)
        acc_train[i,:] = np.array(acc_train_list)
        cost_val[i,:] = np.array(cost_val_list)
        acc_val[i,:] = np.array(acc_val_list)
        cost_test[i,:] = np.array(cost_test_list)
        acc_test[i,:] = np.array(acc_test_list)
        final_acc[i] = final_acc_value
    return cost_train, acc_train, cost_val, acc_val, cost_test, acc_test, np.mean(final_acc)

#Main training procejures for One_run
def OneRun(minivan, convertible, lr, M, num_PC):
    # define learning rate and total number of epochs
    # select train/val/test from class1=Minivan(M)
    trainM, valM, testM = split_dataset_one_run(minivan)
    # select train/val/test from class2=Convertible(C)
    trainC, valC, testC = split_dataset_one_run(convertible)

    return OneRunKernel(0, lr, M, num_PC, trainM, valM, testM, trainC, valC, testC)  #the first parameter in this case is useless

# Kernel function for training during each trail
def OneRunKernel(i, lr, M, num_PC, trainM, valM, testM, trainC, valC, testC, CrossValid = False):
    # combine the datasets
    train1 = np.concatenate((trainM, trainC),axis = 0) # images
    val1 = np.concatenate((valM, valC),axis = 0)
    test1 = np.concatenate((testM, testC),axis = 0)
    # generate input y from the datasets
    train1_y = generate_y(trainM,trainC)
    val1_y = generate_y(valM,valC)
    test1_y = generate_y(testM,testC)

    # Perform PCA on the flattened training set
    # Try using different numbers of components
    projected, mean_image, top_sqrt_eigen_values, top_eigen_vectors, eigen_vectors = PCA(train1,num_PC) # PCA performed only on the training set
    # Check images for top 4 PCs
    PC4_plot(eigen_vectors, i, CrossValid)

    # Check projections - should be zero mean and unit std
    # Answer: this is a good idea because normalization help speed up the gradient descent etc.
    # (Normailzation is necessary for gradient descent) Or the learnt answer will be biased
    # print(np.average(projected,axis=0)) # this is close to zero
    # print(np.std(projected,axis=0)) # !=1 ???

    # generate input x from the datasets
    # Using PCA to reduce the dimensionality (60000 -> 10, which greatly reduce the size of NN)
    train1_x = np.insert(projected, 0, 1, axis = 1) # adding x0 = 1 (a ones column)
    val1_x = apply_PCA(val1,mean_image,top_eigen_vectors,top_sqrt_eigen_values)
    test1_x = apply_PCA(test1,mean_image,top_eigen_vectors,top_sqrt_eigen_values)
    # print(train1_x.shape)
    # print(train1_y.shape)
    # print(w.shape)

    # initial weight vector
    w = theta(train1_x)
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
        w = stepwise_gradient(train1_x,train1_y,w,lr)
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
    val, idx = min((val, idx) for (idx, val) in enumerate(cost_val_list))  #Access the index associated with the lowest loss on the ValidSet
    # print(cost_test_list[idx],acc_test_list[idx])

    return cost_train_list, acc_train_list, cost_val_list, acc_val_list, cost_test_list, acc_test_list, acc_test_list[idx]