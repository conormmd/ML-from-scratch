import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from functionsMultiClass import *
from scipy import optimize

data = loadmat('./3-MultiClass/ex3data1.mat')
X,y = data['X'], data['y'].ravel()
y[y==10] = 0
m=y.size

weights = loadmat('./3-MultiClass/ex3weights.mat')

indices = np.random.permutation(m)
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]
#displayData(sel); plt.show()

input_layer_size = 20*20; hidden_layer_size = 25; num_labels = 10
Theta1, Theta2 = weights['Theta1'], weights['Theta2']
Theta2 = np.roll(Theta2, 1, axis=0)

pred = predict(Theta1, Theta2, X)
#print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))

if indices.size > 0:
    i, indices = indices[0], indices[1:]
    displayData(X[i, :], figsize=(4, 4));plt.show()
    pred = predict(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: {}'.format(*pred))
else:
    print('No more images to display!')