import numpy as np
from numpy.lib.function_base import gradient
from functionsLinRegress import *
data = np.genfromtxt('./1-Regression/data2.txt', delimiter=',')    #Loading training samples
X = data[:,0:2]; y = data[:,2]      #Taking out data
m=len(y)
y=np.reshape(y,(m,1))       #Check y is in shape (m,1) not (m,)

#plotScatter(X[:,0],y);plt.show()

X, mu, sigma = featureNormalisiation(X)     #Feature normalisation of X
X=np.c_[np.ones(m),X]       #Adding a column of 1s
theta = np.zeros((3,1))     #Initilising theta

num_iter=1500       #Iterations for grad descent
alpha=0.01          #Learning rate parameter

theta, J_hist = gradientDescent(X,y,theta,alpha,num_iter)       #Gradient descent of cost function

plotLine(np.arange(0,num_iter),J_hist);plt.show()     #Plotting the cost function over iterations

plotLine(X[:,1],np.matmul(X,theta));plotScatter(X[:,1],y);plt.show()

X = (np.array([1650,3]) - mu)/sigma     #1650 sqr ft & 3 bed house to test, being normalised
X = np.r_[np.ones(1),X]     #Adding bias along row
print(np.matmul(X,theta))   #Calculating cost

