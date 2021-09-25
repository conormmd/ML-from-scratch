import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def displayData(X, example_width=None, figsize=(10, 10)):
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')

def sigmoid(z):
    g=np.zeros((np.shape(z)))
    g=1/(1+np.exp(-z))
    return g

def lrCostFunction(theta, X, y, lambda_):
    m=y.size
    if y.dtype == bool:
        y = y.astype(int)
    J=0
    grad = np.zeros_like(theta)

    h = sigmoid(X@theta.T)

    reg = theta
    reg[0] = 0

    J = (1 / m) * np.sum(-y@np.log(h) - (1 - y)@np.log(1 - h))
    J = J + (lambda_/(2*m))*np.sum(np.square(reg))
    grad = (1 / m) * ((h - y)@X)
    grad = grad + (lambda_/m)*reg

    return J, grad

def oneVsAll(X, y, num_labels, lambda_):
    m,n = X.shape
    all_theta = np.zeros((num_labels, n+1))

    X = np.concatenate([np.ones((m,1)),X], axis=1)

    for c in range(num_labels):
        initial_theta = np.zeros(n+1)
        options = {'maxiter': 50}
        res = optimize.minimize(lrCostFunction, 
                                initial_theta, 
                                (X, (y == c), lambda_), 
                                jac=True, 
                                method='CG',
                                options=options)
        all_theta[c] = res.x
    
    return all_theta

def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]

    p = np.zeros(m)

    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    p = np.argmax(sigmoid(X@all_theta.T), axis = 1)

    return p

def predict(Theta1, Theta2, X):
    if X.ndim == 1:
        X=X[None]
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    p = np.zeros(m)

    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    a2 = sigmoid(X@Theta1.T)
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)

    p = np.argmax(sigmoid(a2@Theta2.T), axis=1)

    return p 
    
