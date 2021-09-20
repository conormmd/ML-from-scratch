import numpy as np
from matplotlib import pyplot
from scipy import optimize
data = np.genfromtxt('./2-LogRegress/data1.txt', delimiter=',')   
X = data[:,0:2]; y = data[:,2];m,n=np.shape(X)

def sigmoid(z):
    g=np.zeros((np.shape(z)))
    g=1/(1+np.exp(-z))
    return g

m, n = X.shape
X=np.c_[np.ones(m),X]

def costFunction(theta, X, y):
    m = len(y)  # number of training examples
    J = 0
    grad = np.zeros_like(theta)
    h = sigmoid(X@theta.T)
    J = (1 / m) * np.sum(-y@np.log(h) - (1 - y)@np.log(1 - h))
    grad = (1 / m) * ((h - y)@X)
    return J, grad


initial_theta = np.zeros(n+1)

cost, grad = costFunction(initial_theta, X, y)

options= {'maxiter': 400}

res = optimize.minimize(costFunction,
                        initial_theta,
                        (X, y),
                        jac=True,
                        method='TNC',
                        options=options)
cost=res.fun
theta=res.x

print(cost, theta)