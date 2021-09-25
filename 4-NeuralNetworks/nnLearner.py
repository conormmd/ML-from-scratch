import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from functionsNeuralNetworks import *
from scipy import optimize

data = loadmat('./4-NeuralNetworks/ex4data1.mat')
X,y = data['X'], data['y'].ravel()
y[y==10] = 0; m = y.size

rand_indices = np.random.choice(m, 100, replace = False)
sel = X[rand_indices, :]
#displayData(sel); plt.show()

weights = loadmat('./3-MultiClass/ex3weights.mat')
input_layer_size = 20*20; hidden_layer_size = 25; num_labels = 10
Theta1, Theta2 = weights['Theta1'], weights['Theta2']
Theta2 = np.roll(Theta2, 1, axis=0)

nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)

options = {'maxiter': 100}
lambda_ = 1

costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                        hidden_layer_size,
                                        num_labels, X, y, lambda_)
res = optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

nn_params = res.x

Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))


pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))

displayData(Theta1[:, 1:]);plt.show()