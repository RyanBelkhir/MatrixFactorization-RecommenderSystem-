import numpy as np
from numpy import genfromtxt
import pandas as pd
import csv
import time
import math
import random
import numba as nb
import sklearn
from sklearn.cluster import KMeans
from sklearn.utils import extmath
from sklearn import preprocessing
from scipy import linalg
from data_init import U, M, R, R_train, R_test
from optimizer import SGD, GradientDescent, ALS, SVD
from optimizerClusterize import SGD_clusterize, ALS_clusterize, GD_clusterize, SVD_clusterize
import matplotlib.pyplot as plt


if __name__=='__main__':
    #Hyperparameters settings
    K = 6

    als = ALS(K, U, M, R_train, R_test)
    U_als, M_als = als.fit()
    als.plot_history()
    
    als_clusterize = ALS_clusterize(K, "users", R_train, R_test)
    als_clusterize.fit(U,M)
    als_clusterize.plot_history()

    #K = range(1,15)
    #BICs = []
    
    #for k in K:
    #    print("hello")
    #    max_ll = 0
    #    lls = []
    #    for s in range(5):
    #        mixture, post  = gm.init(R_train, K=k, seed=s)
    #        new_mixture, new_post, new_ll = em.run(R_train, mixture, post)
    #        lls.append(new_ll)
    #    
    #   max_ll = max(lls)
    #    BICs.append((max_ll, lls.index(max_ll)))
        
    #print(BICs)
    
    #mixture, post = gm.init(R_train, K=11, seed=0)
    #new_mixture, new_post, new_ll = em.run(R_train, mixture, post)
    #R_hat = em.fill_matrix(R_train, new_mixture)
    #print(rmse(R_test,R_hat))
    