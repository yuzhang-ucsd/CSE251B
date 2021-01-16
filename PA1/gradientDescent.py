import numpy as np

# define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# define initial theta
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

# define one round of gradient descent and weights update
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

# define prediction accuracy
def predict_acc(X,y,theta):
    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    prob = sigmoid(X * theta.T)
    prediction_list =  [0 if x < 0.5 else 1 for x in prob] # prediction=1 when => 0.5
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(prediction_list, y)] # compare prediction and actual y
    accuracy = sum(correct)/len(correct)*100 # accuracy in %
    return accuracy


