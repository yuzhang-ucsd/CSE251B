import matplotlib.pyplot as plt
import os, random
import numpy as np
from PIL import Image

# Convert 2-D image(200x300) to 1-D vector
def flatten_img(imgset):
    new_set = []
    for i in range(len(imgset)):
        new_set.append(imgset[i].flatten())
    return np.array(new_set)

# Split the dataset into 8/10 portion for training, 1 set for validation and 1 set for testing
# Repeat k times and then average
def split_dataset_k_fold(dataset, k):
    size = len(dataset)
    cut_train = int((10-2)/10 * size) # training cutoff
    cut_val = int(1/10 * size) # validation cutoff
    train = []
    val = []
    test = []
    for i in range(k):
        # shuffle the dataset
        idx = np.arange(0,size)
        random.shuffle(idx)
        shuffled_M = dataset[idx]
        # select subsets of data
        train_tmp = shuffled_M[:cut_train] # (k-2) training set
        val_tmp = shuffled_M[cut_train:(cut_train+cut_val)] # 1 validation set
        test_tmp = shuffled_M[(cut_train+cut_val):] # 1 testing set
        train.append(train_tmp)
        val.append(val_tmp)
        test.append(test_tmp)
    return train, val, test


# Split the dataset into 8/10 portion for training, 1 set for validation and 1 set for testing
def split_dataset_one_run(dataset):
    size = len(dataset)
    cut_train = int((10-2)/10 * size) # training cutoff
    cut_val = int(1/10 * size) # validation cutoff
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
def PC4_plot(eigen_vectors, i, CrossValid):
    imgs = []
    for n in range(4):
        vh = eigen_vectors[n]
        vh_img = np.reshape(vh,(200,300))
        img = Image.fromarray(vh_img)
        imgs.append(img)
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
    if CrossValid:
        plt.savefig('./figures/Q5c_top4PCs_' + str(i) + '.png')
    else:
        plt.savefig('./figures/Q5b_top4PCs.png')

# Generate y vectors (class=0/1 = Con/Min) for train/val/test set
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

def plotFunc(cost_list, accuracy_list, SetName = 'TrainSet', do_save_fig = False, CrossValid = False, Epoch =  0, Interval = 0):
    fig = plt.figure()
    if CrossValid:  #k sets experiments
        cost_bar = np.std(cost_list, axis = 0)
        accu_bar = np.std(accuracy_list, axis = 0)
        Bar_number = int((Epoch - 1) / Interval + 1)
        I = np.linspace(0, Epoch - 1, Bar_number)
        I = [int(i) for i in I] #round to integer
        cost_list = np.mean(cost_list, axis = 0)
        accuracy_list = np.mean(accuracy_list, axis = 0)
        plt.errorbar(I, cost_list[I], yerr = cost_bar[I], fmt = '.b', capsize=5)
        plt.errorbar(I, accuracy_list[I], yerr = accu_bar[I], fmt = '.r', capsize=5)
    Costlabel = SetName + ' Error'
    Accuracylabel = SetName + ' Error'
    plt.plot(cost_list, 'b', label = Costlabel)
    plt.plot(accuracy_list, 'r', label = Accuracylabel)
    plt.xlabel('M epochs')
    plt.ylabel('Cost')
    plt.title('Loss and Accurcy in one run on ' + SetName)
    # ax.grid(True)
    plt.grid('color')
    plt.legend(['Loss', 'Accuracy'])
    if do_save_fig:
        if CrossValid:
            plt.savefig('./figures/Q5c_' + SetName + '_curves.png')
        else:
            plt.savefig('./figures/Q5b_' + SetName + '_curves.png')
