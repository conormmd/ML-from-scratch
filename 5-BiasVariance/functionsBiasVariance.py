import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def linearRegCostFunction(X, y, theta, lambda_=0):

    m = y.size
    J = 0
    grad = np.zeros_like(theta)

    h = X @ theta
    J = (1/ (2*m)) * np.sum(np.square(h - y)) + (lambda_/ (2*m)) * np.sum(np.square(theta[1:]))
    grad = (1/m) * (h-y) @ X
    grad[1:] = grad[1:] + (lambda_/m) * theta[1:]

    return J, grad

def trainLinearReg(linearRegCostFunction, X, y, lambda_=0, maxiter=200):
    initial_theta = np.zeros(X.shape[1])

    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)

    options = {'maxiter':maxiter}

    res = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)

    return res.x

def learningCurve(X, y, Xval, yval, lambda_=0):
    m = y.size
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(m):
        
        Xtrain = X[0:i+1,:]     #i+1 to prevent zero error fitting (ie one point regression)?
        ytrain = y[0:i+1]
        theta_t = trainLinearReg(linearRegCostFunction, Xtrain, ytrain, lambda_=lambda_)

        error_train[i], _ = linearRegCostFunction(Xtrain, ytrain, theta_t, lambda_=0)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta_t, lambda_=0)
    
    return error_train, error_val

def polyFeatures(X, p):
    X_poly = np.zeros((X.shape[0], p))

    for i in range(p):
        X_poly[:, i] = X[:, 0]**(i+1)

    return X_poly

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm/sigma

    return X_norm, mu, sigma

def validationCurve(X, y, Xval, yval):
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    n = len(lambda_vec)

    error_train = np.zeros(n)
    error_val = np.zeros(n)

    for i in range(n):
        lambda_try = lambda_vec[i]
        theta_t = trainLinearReg(linearRegCostFunction, X, y, lambda_=lambda_try)
        error_train[i], _ = linearRegCostFunction(X, y, theta_t, lambda_=0)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta_t, lambda_=0)
    
    return lambda_vec, error_train, error_val


def plotFit(polyFeatures, min_x, max_x, mu, sigma, theta, p):
    """
    Plots a learned polynomial regression fit over an existing figure.
    Also works with linear regression.
    Plots the learned polynomial fit with power p and feature normalization (mu, sigma).
    Parameters
    ----------
    polyFeatures : func
        A function which generators polynomial features from a single feature.
    min_x : float
        The minimum value for the feature.
    max_x : float
        The maximum value for the feature.
    mu : float
        The mean feature value over the training dataset.
    sigma : float
        The feature standard deviation of the training dataset.
    theta : array_like
        The parameters for the trained polynomial linear regression.
    p : int
        The polynomial order.
    """
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)

    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly -= mu
    X_poly /= sigma

    # Add ones
    X_poly = np.concatenate([np.ones((x.shape[0], 1)), X_poly], axis=1)

    # Plot
    plt.plot(x, np.dot(X_poly, theta), '--', lw=2)


