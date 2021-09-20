from math import cos
import numpy as np
from scipy import optimize
from functionsLogRegress import *
data = np.genfromtxt('./2-LogRegress/data1.txt', delimiter=',')   
X = data[:,0:2]; y = data[:,2];m,n=np.shape(X)
pos = (y == 1); neg = (y == 0)      #Bool arrays for data sets (X,y) with index of positive and negative y values

plotScatter(X[pos,0],X[pos,1],'blue'); plotScatter(X[neg,0],X[neg,1],'red')     #Using bool arrays to plot exclusively 1 or 0 values for y

X=np.c_[np.ones(m),X]
initial_theta=np.zeros(n+1)

options = {'maxiter':400}       #Telling scipy optimize how many iterations to run

res = optimize.minimize(costFunction,       #Our cost function being used
                        initial_theta,      #Initial values for theta
                        (X, y),             #Data being fitted against
                        jac=True,           #Return jacobian (gradient)
                        method='TNC',       #Truncated Newton algorithm
                        options=options)    #Using our chosen number of max iterations

theta=res.x; cost=res.fun

plotDecisionBoundary(plotData,theta,X,y);plt.show()
p=predict(theta,X)
print(np.mean(p==y))