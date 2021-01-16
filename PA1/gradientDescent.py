import numpy as np

# define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# initialize theta
def theta(X):
    theta = np.zeros(len(X[0])) # if the input X is Mxd
    return theta

# define the cost function given training examples and theta
def cost_function(X,y,theta):
    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    y = np.asmatrix(y)
    cost_i = np.multiply(-y, np.log(sigmoid(X * theta.T))) - np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    cost = np.sum(cost_i) / (len(X))
    return cost

#
def stepwise_gradient(X,y,theta,lr):
    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    y = np.asmatrix(y)
    error = sigmoid(X * theta.T) - y
    grad = X.T * error / len(X)
    grad = np.squeeze(np.asarray(grad))
    theta_updated = theta - lr*grad
    theta_updated = np.squeeze(np.asarray(theta_updated))
    return grad, theta_updated





