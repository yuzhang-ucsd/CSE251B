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
