import numpy as np
from numpy.lib.function_base import gradient
from functionsLinRegress import *
X,y = np.genfromtxt('./1-LinRegress/data1.txt', delimiter=',').T    #Loading training samples
m=len(y)    #Num of training samples
#y=np.reshape(y,(m,1));X=np.reshape(X,(m,1))       #Check X,y is in shape (m,1) not (m,)

plotScatter(X,y,'red');plt.show()       #Plotting training data

X=np.c_[np.ones(m),X]      #Adding a column of 1s
theta=np.array([[-1],[2]])       #Initilising theta

num_iter=1500       #Iterations for grad descent
alpha=0.01          #Learning rate parameter

theta, J_hist = gradientDescent(X,y,theta,alpha,num_iter)       #Gradient descent of cost function

plotLine(np.arange(0,num_iter),J_hist);plt.show()     #Plotting the cost function over iterations
plotLine(X[:,1],np.matmul(X,theta));plotScatter(X[:,1],y,'red');plt.show()        #Plotting the fitted regression


