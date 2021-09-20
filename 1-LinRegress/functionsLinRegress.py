import numpy as np
import matplotlib.pyplot as plt

def plotScatter(x,y,colour):
    plt.scatter(x,y,marker='x',c=colour)
    return

def computeCost(X,y,theta):
    m=len(y)
    y=np.reshape(y,(m,1))
    J=0
    least_square = (np.matmul(X,theta)-y)**2
    J=1/(2*m)*np.sum(least_square)
    return J

def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)
    J_hist = np.zeros((num_iters,1))
    y=np.reshape(y,(m,1))
    for i in range(num_iters):
        theta = theta - (alpha/m) * (np.matmul(X.T,((np.matmul(X,theta))-y)))
        J_hist[i] = computeCost(X,y,theta)
    return theta, J_hist

def plotLine(x,y):
    plt.plot(x,y)
    return

def featureNormalisiation(X):
    X_norm = X
    mu = np.zeros((1, np.shape(X)[1]))
    sigma = np.zeros((1, np.shape(X)[1]))

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma

def normalEquation(X,y):
    theta = np.zeros()