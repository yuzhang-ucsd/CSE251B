import matplotlib.pyplot as plt
import os, random
import numpy as np

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
