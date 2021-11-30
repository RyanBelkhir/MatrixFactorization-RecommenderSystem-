from typing import NamedTuple, Tuple
import numpy as np


class GaussianMixture(NamedTuple):
    """Tuple holding a gaussian mixture"""
    mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component


def init(R: np.ndarray, K: int,
         seed: int = 0) -> Tuple[GaussianMixture, np.ndarray]:
    """Initializes the mixture model with random points as initial
    means and uniform assingments"""
    
    np.random.seed(seed)
    n, _ = R.shape
    p = np.ones(K) / K

    # select K random points as initial means
    mu = R[np.random.choice(n, K, replace=False)]
    var = np.zeros(K)
    # Compute variance
    for j in range(K):
        var[j] = ((R - mu[j])**2).mean()

    mixture = GaussianMixture(mu, var, p)
    post = np.ones((n, K)) / K

    return mixture, post


def rmse(X, Y):
    return np.sqrt(np.mean((X - Y)**2))

def bic(R: np.ndarray, mixture: GaussianMixture,
        log_likelihood: float) -> float:
    """Computes the Bayesian Information Criterion for a
    mixture of gaussians"""
    
    n, _ = R.shape
    mu = mixture.mu
    K, d = mu.shape
    p = K*(d+2) - 1
    return log_likelihood - 0.5*p*np.log(n)

def cluster_assignment(post: np.ndarray):
    """Compute the assignment for each cluster with the 
    probability of belonging to this cluster"""
    
    n, K = post.shape
    
    assignments = {j: [] for j in range(K)}
    
    for i in range(n):
        for j in assignments.keys():
            if post[i,j] != 0:
                assignments[j].append(i)
                
    return assignments
