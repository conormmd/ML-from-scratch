import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from functionsBiasVariance import *
from scipy import optimize

data = loadmat('./5-BiasVariance/ex5data1.mat')
X, y = data['X'], data['y'][:,0]
Xtest, ytest = data['Xtest'], data['ytest'][:,0]
Xval, yval = data['Xval'], data['yval'][:,0]

m = y.size

#plt.plot(X, y, 'ro', ms=10, mec='k', mew=1); plt.xlabel('Change in water level (x)'); plt.ylabel('Water flowing out of the dam (y)'); plt.show()

X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)
Xval_aug = np.concatenate([np.ones((yval.size, 1)), Xval], axis=1)
error_train, error_val = learningCurve(X_aug, y, Xval_aug, yval, lambda_=0)

plt.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2);plt.title('Learning curve for linear regression');plt.legend(['Train', 'Cross Validation']);plt.xlabel('Number of training examples');plt.ylabel('Error');plt.axis([0, 13, 0, 150]); plt.show()

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = np.concatenate([np.ones((m, 1)), X_poly], axis=1)

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.concatenate([np.ones((ytest.size, 1)), X_poly_test], axis=1)

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.concatenate([np.ones((yval.size, 1)), X_poly_val], axis=1)

lambda_ = 100 
theta = trainLinearReg(linearRegCostFunction, X_poly, y,
                             lambda_=lambda_, maxiter=55)

# Plot training data and fit
plt.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')
plotFit(polyFeatures, np.min(X), np.max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)');plt.ylabel('Water flowing out of the dam (y)');plt.title('Polynomial Regression Fit (lambda = %f)' % lambda_);plt.ylim([-20, 50]);plt.show()

plt.figure()
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
plt.plot(np.arange(1, 1+m), error_train, np.arange(1, 1+m), error_val)
plt.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_);plt.xlabel('Number of training examples');plt.ylabel('Error');plt.axis([0, 13, 0, 100]);plt.legend(['Train', 'Cross Validation']);plt.show()

print('Polynomial Regression (lambda = %f)\n' % lambda_)
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))


lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

plt.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
plt.legend(['Train', 'Cross Validation']);plt.xlabel('lambda');plt.ylabel('Error');plt.show()

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
    print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))