"""
Gaussian Class adapted from 
Tubingen University's Probabilistic Machine Learning Course
"""

import jax
import jax.numpy as jnp
import functools
import dataclasses

@dataclasses.dataclass
class Gaussian:
    mu: jnp.ndarray
    Sigma: jnp.ndarray

    def sample(self, key, n):
        return jax.random.multivariate_normal(key, self.mu, self.Sigma, (n,), method='svd')
    
    @functools.cached_property
    def L(self):
        return jnp.linalg.cholesky(self.Sigma)
      
    @functools.cached_property
    def L_factor(self):
        return jax.scipy.linalg.cho_factor(self.Sigma, lower=True)
    
    @functools.cached_property
    def dim(self):
        return self.mu.shape[0]
    
    @functools.cached_property
    def precision(self):
        return jnp.linalg.inv(self.Sigma)
    
    def prec_mult(self, x):
        return jax.scipy.linalg.cho_solve(self.L_factor, x)
      
    @functools.cached_property
    def pmu(self):
        return self.prec_mult(self.mu)
    
    @functools.cached_property
    def logdet(self):
        return 2 * jnp.sum(jnp.log(jnp.diag(self.L)))
      
    def logpdf(self, x):
        return -0.5 * (self.dim * jnp.log(2 * jnp.pi) 
                       + self.logdet
                       + (x - self.mu).T @ self.prec_mult(x - self.mu))
    def pdf(self, x):
        return jnp.exp(self.logpdf(x))
      
    def __mul__(self, other):
      Sigma = jnp.linalg.inv(self.precision + other.precision)
      mu = Sigma @ (self.pmu + other.pmu)
      return Gaussian(mu, Sigma)
    
    def __rmatmul__(self, A: jnp.ndarray):
      return Gaussian(A @ self.mu, A @ self.Sigma @ A.T)
  
    @functools.singledispatchmethod
    def __add__(self, other: jnp.ndarray):
      other = jnp.asarray(other)
      return Gaussian(self.mu + other, self.Sigma)
    
    def condition(self, A, y, Lambda):
      Gram = A @ self.Sigma @ A.T + Lambda
      L = jax.scipy.linalg.cho_factor(Gram, lower=True)
      mu = self.mu + self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(L, y - A @ self.mu)
      Sigma = self.Sigma - self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(L, A @ self.Sigma)
      return Gaussian(mu, Sigma)
  
    @functools.cached_property
    def std(self):
        return jnp.sqrt(jnp.diag(self.Sigma))
    
@Gaussian.__add__.register
def _add_gaussian(self, other: Gaussian):
    return Gaussian(self.mu + other.mu, self.Sigma + other.Sigma)