from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from gm import GaussianMixture

def log_gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:
    """Computes the log probablity of vector x under a normal distribution"""
    
    d = len(x)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((x - mean)**2).sum() / var
    return log_prob

def estep(R: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component"""
    
    n, _ = R.shape
    mu, var, p = mixture
    K, _ = mu.shape
    post = np.zeros((n,K)) 
    ll = 0
    
    for i in range(n):
        mask = (R[i,:] != 0)
        
        for j in range(K):
            log_likelihood = log_gaussian(R[i,mask], mu[j, mask], var[j])
            post[i,j] = np.log(p[j] + 1e-16) + log_likelihood
        
        trick = logsumexp(post[i,:])
        post[i,:] = post[i,:] - trick
        ll += trick
        
    return np.exp(post), ll


def mstep(R: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset"""
    
    n, d = R.shape
    mu = mixture.mu.copy()
    K, _ = mu.shape
    var = np.zeros(K)
    
    n_hat = post.sum(axis=0)
    p = n_hat/n
    
    for j in range(K):
        sse = 0
        weight = 0
        for l in range(d):
            mask = (R[:,l] != 0)
            n_sum = post[mask,j].sum()
            if n_sum > 0:
                #update mean
                mu[j,l] = (R[mask,l] @ post[mask,j])/n_sum
                
            #compute variance

            sse += ((mu[j,l] - R[mask,l])**2) @ post[mask, j]
            weight += n_sum
            #print(weight)
        
        var[j] = sse/weight
        if var[j] < min_variance:
            var[j] = min_variance
        
    return GaussianMixture(mu, var, p)


def run(R: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model"""
    
    old_ll = None
    new_ll = None
    
    while(old_ll is None or (new_ll - old_ll) > 1e-6*abs(new_ll)):
        
        old_ll = new_ll
        post, new_ll = estep(R, mixture)
        mixture = mstep(R, post, mixture)
        
    return mixture, post, new_ll


def fill_matrix(R: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model"""
    
    n, d = R.shape
    R_hat = R.copy()
    K, _ = mixture.mu.shape

    for i in range(n):
        mask = (R[i, :] != 0)
        mask0 = (R[i, :] == 0)
        post = np.zeros(K)
        for j in range(K):
            log_likelihood = log_gaussian(R[i, mask], mixture.mu[j, mask],
                                          mixture.var[j])
            post[j] = np.log(mixture.p[j]) + log_likelihood
        post = np.exp(post - logsumexp(post))
        R_hat[i, mask0] = post @ mixture.mu[:, mask0]
    
    return R_hat



        
    
