import numpy as np
import matplotlib.pyplot as plt

def ML_feature_scaling(X):
    """
    Normalizes the features in X subtracting the mean and dividng by the standard deviation.

    Parameters
    ---------
    X: array like. Shape (m x n)
    
    Returns
    ---------
    X_norm: normalized array like. Shape (m x n)
    mu: feature's mean
    sigma: feature's standard deviation 
    """
    X_norm = X.copy()

    cols = np.shape(X)[-1]
    mu = np.zeros(cols)
    sigma = np.zeros(cols)
    
    for i in range(cols):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])
    
        X_norm[:, i] = (X[:, i] - mu[i])/sigma[i]
    
    return X_norm, mu, sigma

def ML_LinearReg_CostFunc_Multi(X,y,theta):
    """
    Compute the cost fuction for linear regression with multiple variables

    Equantion implemented
    ---------
    J(theta) = frac{1}{2m} (X theta - y)^T (X theta -y)

    Parameters
    ---------
    X: array like. Shape (m x n+1)
    y: array like. Shape (m)
    theta: array like. Shape (n+1, )

    Returns
    ---------
    J: float. Cost function value
    """
    m=y.shape[0]

    J=0
    J = 1/(2*m)*(np.dot((np.dot(X, theta) - y).T, np.dot(X, theta) - y))
    
    return J    


def ML_GradDescent_Multi(X,y,theta,alpha,num_iterations):
    """
    Compute gradient descent to learn theta

    Equantion implemented
    ---------
    theta = theta - alpha frac{1}{m} (X^T (X theta -y))

    Parameters
    ---------
    X: array like. Shape (m x n+1)
    y: array like. Shape (m)
    theta: array like. Shape (n+1,)
    alpha: float. Learning rate
    num_iterations: int. Number of iterations to run GD

    Returns
    ---------
    theta: array like. The learned linear regression parameters. Shape (n+1,)
    J_history: list. Values of the cost function for each iteration
    """
    m=y.shape[0]
    theta=theta.copy()
    J_history=[]
    
    for i in range(num_iters):
        theta = theta - alpha*(1/m)*(np.dot(X.T,np.dot(X,theta)-y))
        J_history.append(computeCostMulti(X,y,theta))
        
    return theta,J_history

def ML_NormalEq(X,y):
    """
    Compute analytically the linear regression model.
    Attentiom: it works well for small n (approx 10^3)

    Equantion implemented
    ---------
    theta = (X^T X)^{-1} X^T y


    Parameters
    ---------
    X: array like. Shape (m x n+1)
    y: array like. Shape (m, )

    Returns
    ---------
    theta: array like. Estimated linear regression parameters. Shape (n+1,)
    """
    theta=np.zeros(X.shape[1])
    
    matrix_inv=np.linalg.pinv(np.dot(X.T,X))
    theta = np.dot(np.dot(matrix_inv,X.T),y)
    
    return theta