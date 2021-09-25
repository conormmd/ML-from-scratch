import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from functionsMultiClass import *
from scipy import optimize

input_layer_size = 20*20; num_labels = 10

data = loadmat('./3-MultiClass/ex3data1.mat')
X,y = data['X'], data['y'].ravel()
y[y==10] = 0
m=y.size

rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]
#displayData(sel); plt.show()

theta_t = np.array([-2,-1,1,2], dtype=float)
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)


lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)

pred = predictOneVsAll(all_theta, X); print('Training Set Accuracy:', np.mean(pred == y) * 100)


