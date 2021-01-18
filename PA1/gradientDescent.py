import numpy as np

## Softmax
# define softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 1)

# define initial weights for softmax function
def theta_softmax(X, NumCat):
    theta = np.zeros([len(X[0]), NumCat])
    return theta

# define the corss-entropy loss function for multiple categories
def cost_function_softmax(X,y,theta):
    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    y = np.asmatrix(y)
    a_k = X * theta
    y_k = softmax(a_k)
    cost = -np.multiply(y, np.log(y_k)) #multiply in an element-wise way
    cost = np.sum(cost) / (cost.shape[0] * cost.shape[1]) #average over the number of training examples and the number of categories
    return cost

def predict_acc_softmax(X,y,theta):
    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    a_k = X * theta
    y_k = softmax(a_k)
    category = np.argmax(y, axis = 1)
    prediction = np.argmax(y_k, axis = 1)
    correct = [1 if a == b else 0 for (a, b) in zip(category, prediction)] # compare prediction and actual y
    accuracy = sum(correct) / len(correct)
    return accuracy

# define one round of gradient descent and weights update
def stepwise_gradient_softmax(X,y,theta,lr):
    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    y = np.asmatrix(y)
    #update (batch version)
    a_k = X * theta
    y_k = softmax(a_k)
    error = y_k - y
    grad = X.T * error / X.shape[0]
    grad = np.squeeze(np.asarray(grad))
    theta_updated = theta - lr * grad # update weights using some learning rate times gradient
    # theta_updated = np.squeeze(np.asarray(theta_updated))
    return theta_updated

    # for i in range(theta.shape[2]):

# Generate the Confusion Matrix on test set results
def GeneConMatrix(X, y, theta):
    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    a_k = X * theta
    y_k = softmax(a_k)
    category = np.argmax(y, axis = 1)
    prediction = np.argmax(y_k, axis = 1)
    Confusion_matrix = np.zeros([4,4])
    for i in range(len(category)):
        Confusion_matrix[category[i], prediction[i]] += 1
    return Confusion_matrix

## Logistic
# define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# define initial weights -> theta
def theta(X):
    theta = np.zeros(len(X[0])) # if the input X is Mxd
    return theta

# define the corss-entropy loss function given training examples and theta
def cost_function(X,y,theta):
    """
    Args:
    	X: has shape Mxd' where M is the number of images and d' is the reduced dimension of each image after applying PCA
    	y: has shape Mx1 where M is the number of images and for each image 1 is one class and 0 is the other
    	theta: weight vector, has shape d'x1 and d' is the reduced dimension of each image after applying PCA

    Returns:
    	cost: corss-entropy loss of the given weights (theta)
    """
    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    y = np.asmatrix(y)
    cost_i = np.multiply(-y, np.log(sigmoid(X * theta.T))) - np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T))) #multiply in an element-wise way
    cost = np.sum(cost_i) / (len(X))
    return cost

# define one round of gradient descent and weights update
def stepwise_gradient(X,y,theta,lr):
    """
    Args:
    	X: has shape Mxd' where M is the number of images and d' is the reduced dimension of each image after applying PCA
    	y: has shape Mx1 where M is the number of images and for each image 1 is one class and 0 is the other
    	theta: weight vector, has shape d'x1 and d' is the reduced dimension of each image after applying PCA
    	lr: learning rate

    Returns:
    	theta_updated: updated weight vector
    """
    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    y = np.asmatrix(y)
    error = sigmoid(X * theta.T) - y
    grad = X.T * error / len(X)
    grad = np.squeeze(np.asarray(grad))
    theta_updated = theta - lr*grad # update weights using some learning rate times gradient
    theta_updated = np.squeeze(np.asarray(theta_updated))
    return theta_updated

# define prediction accuracy
def predict_acc(X,y,theta):
    """
    Args:
    	X: has shape Mxd' where M is the number of images and d' is the reduced dimension of each image after applying PCA
    	y: has shape Mx1 where M is the number of images and for each image 1 is one class and 0 is the other
    	theta: weight vector, has shape d'x1 and d' is the reduced dimension of each image after applying PCA

    Returns:
    	accuracy: percent correct when choosing the category according to the rule (=>0.5 if class1)
    """
    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    prob = sigmoid(X * theta.T)
    prediction_list =  [0 if x < 0.5 else 1 for x in prob] # prediction=1 when => 0.5 (probability of x belonging to C_1)
    correct = [1 if a == b else 0 for (a, b) in zip(prediction_list, y)] # compare prediction and actual y
    accuracy = sum(correct)/len(correct) # accuracy in %
    return accuracy

