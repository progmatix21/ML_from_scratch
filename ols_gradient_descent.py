#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 11:30:32 2023

@author: github/progmatix21
"""

from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

def ols_gd(X,y, alpha=0.001, n_iter=50):
    '''
    X: feature matrix(mxn), m rows of n features.
    y: response variable array of lenght m
    alpha: the learning rate
    n_iter: number of iterations of gradient descent
    returns weight vector
    '''
    # Insert a column of 1s into X to signify x0 corresponding to w0
    _X = np.insert(X,0,1,axis=1)
    # number of weights is same as number of features
    n_w = _X.shape[1]
    # Create a column vector that includes w0 and intialize to 1s
    w = np.array([0.1]*n_w, dtype=float).reshape(-1,1)
    alpha = alpha  # Initialize the learning rate

    history = list() # history to record convergence of weights
    history.append(w.ravel().copy()) # append the initial weights
    # the gradient descent loop
    for _ in range(n_iter):
        for j in range(w.shape[0]): # for each weight: w is a column vector
            w[j,0] += -alpha*np.sum((np.matmul(_X,w).ravel()-y)*_X[:,j])
        history.append(w.ravel().copy())

    return w.ravel(),history

if __name__ == '__main__':
    # Generate the data
    w0 = 50  # bias is w0
    X,y,c = make_regression(n_samples=100,n_features=1,noise=4,bias=w0,coef=True)

    # Run the regression
    weights, history = ols_gd(X,y,n_iter=50)

    # Plot the convergence of weights
    plt.plot(history, '-')
    plt.legend([f'$w_{i}$' for i in range(len(weights))])
    plt.title('Converging weights')
    plt.grid()
    plt.show()
    # Plot data and trace of the line
    if X.shape[1] == 1:
        plt.scatter(X[:,0],y,c=y,cmap='viridis')
        for ws in history:
            plt.plot(X[:,0], [ws[0]+np.dot(Xrow,ws[1:]) for Xrow in X], color='Blue',alpha=0.1)
        plt.legend(['data','prediction'])
        plt.title('OLS straight line regression')
        plt.show()

    #'''
    # Basis function regression to model a non-linear function
    # Generate the function
    n_points = 100
    rng = np.random.RandomState()
    x = rng.rand(n_points)
    y = np.sin(2*3.14159*x) + 0.1 * rng.randn(n_points)
    plt.scatter(x, y)

    # apply a polynomial basis function to x
    poly_basis = PolynomialFeatures(9, include_bias=False)
    X = poly_basis.fit_transform(x[:, None])
    
    # Run the regression
    weights, history = ols_gd(X,y,n_iter=50000,alpha=0.005)

    # Plot the journey of the curve fitting
    for i, ws in enumerate(history, start=1):
        if i in [1,2,3,4,5,10,100,500,1000,2000,3000,4000,5000, \
                 20000,25000,30000,45000,50000]:   #plot every nth instance of candidate predictions
            plt.scatter(X[:,0], [ws[0]+np.dot(Xrow,ws[1:]) for Xrow in X], s=10, color='Black',alpha=.1)

    plt.legend(['True','Estimated'])
    plt.grid()
    plt.title('OLS basis function regression')    
    plt.show()
    #'''

