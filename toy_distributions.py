
import numpy as np
import math
import bayespy.utils as bayespy
import bayespy.plot as bpplt
from scipy.special import expit, logit


#[]

def toy_gaussian_mixture(N,D,K):

    pi = np.ones((K,1))/K
    mu = 4*np.random.rand(D,K)-2
    var = np.random.uniform(0.05, 0.1, size=(D,K))

    Y = np.zeros((1,D))
    for k in range(K):
        Y = np.vstack((Y,(np.random.multivariate_normal(mu[:,k], np.diag(var[:,k]), int(N*pi[k])))))
        #Y = np.vstack((Y,(np.random.multivariate_normal(mu[k], np.diag(var[k]), int(N*pi[k])))))

    Y = Y[1:,:]
    return Y

def toy_categorical_mixture(N,D,K):

    pi = np.ones((K,1))/K
    #pi = [0,1]
    probs_cat = np.random.rand(D,K)
    #probs_cat = [1, 8.9]
    probs_cat = probs_cat /np.tile(np.sum(probs_cat,0)[np.newaxis,:],(D,1))
    Y = np.zeros((1, D))
    for k in range(K):
        Y = np.vstack((Y, np.random.multinomial(1, probs_cat[:, k], size=int(N * pi[k]))))

    Y = Y[1:, :]
    return Y

def toy_multinomial_mixture(N,D,M,K):

    pi = np.ones((K,1)) / K
    probs = np.random.rand(D,K)  # Probabilities of each of the Dcat outcomes
    #probs = probs/np.tile(np.sum(probs,0)[np.newaxis,:],(D,1))

    Y = np.zeros((1,D))
    for k in range(K):
        Y = np.vstack((Y, np.random.multinomial(M,probs[:,k],size= int(N*pi[k]))))  # Tengo N muestras de valor de 0 a 20 con 3 eventos

    Y = Y[1:, :]
    return Y



def toy_bernoulli_mixture(N,D,K):

    pi = np.ones((K,1)) / K
    mu = np.random.rand(D,K)

    Y = np.zeros((1,D))
    for k in range(K):
        param = np.tile(mu[np.newaxis,:,k], (int(N*pi[k]), 1))
        Y = np.vstack((Y, np.random.binomial(1, param)))

    Y = Y[1:, :]
    return Y


def generate_mixture(K,N,D_real,D_bin):

    # Pi and S
    pi = np.random.dirichlet(np.ones(K))
    pi = [0.25,0.25,0.5]
    print(pi)
    s = bayespy.random.categorical(pi,size=N)
    print(s)

    # Distribution parameters
    L = 3
    z = np.random.multivariate_normal(np.zeros(L),np.eye(L))
    W_c = np.random.rand(K, D_real, L)
    gaussian_covar = [[2, 0, 0 ,0,0],[0, 4, 0 ,0,0],[0, 0, 0.4 ,0,0],[0, 0, 0 ,9.3,0],[0, 0, 0 ,0,0.2]], \
                     [[7, 0, 0, 0, 0], [0, 0.1, 0, 0, 0], [0, 0, 6.5, 0, 0], [0, 0, 0, 0.3, 0], [0, 0, 0, 0, 2]],\
                    [[3, 0, 0, 0, 0], [0, 1.5, 0, 0, 0], [0, 0, 0.5, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, 2.5]]

    # Real Observations
    y_real = np.zeros((N,D_real))
    for i in range(N):
            k = s[i]
            gaussian_means = np.dot(W_c[k,:, :], z)
            y_real[i]= np.random.multivariate_normal(gaussian_means,gaussian_covar[k])

    y_bin =  np.zeros((N,D_bin))
    W_d = np.random.rand(K, D_bin, L)

    for i in range(N):
        k = s[i]
        probs = np.dot(W_d[k,:, :], z)
        y_bin[i]= bayespy.random.bernoulli(expit(probs))

    print(y_real)
    print(y_bin)
    return y_real,y_bin,s
