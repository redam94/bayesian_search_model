"""
File contains classes for GaussianProcesses and ConditionalGaussianProcesses
Modified from Tubingen University's Probabilistic Machine Learning Course
"""
from utils.gaussians import Gaussian
from utils.gp.kernels import Kernel
import matplotlib.colors as colors
import dataclasses
import jax.numpy as jnp
import jax
import functools
import numpy as np
import abc
from tueplots.constants.color import rgb


@dataclasses.dataclass
class GaussianProcess:
  """Gaussian Process Base Class
  Attributes:
    m (callable): mean function for gaussian process
    k (Kernel)
  
  """
  m: callable
  k: Kernel
  def __call__(self, x):
    return Gaussian(mu=self.m(x), Sigma=self.k(x[:, None, :], x[None, :, :]))
  
  def condition(self, X: jnp.ndarray, y: jnp.ndarray, sigma: float):
    return ConditionalGausianProcess(self, y, X, Gaussian(mu=jnp.zeros_like(y), Sigma=sigma*jnp.eye(len(y))))
  
  def plot(self,
           ax,
           x,
           color=rgb.tue_gray,
           yrange=(-3, 3),
           yres=200,
           mean_kwargs={},
           std_kwargs={},
           gamma = .1,
           num_samples=0,
           rng_key=None):
    gp_x = self(x)
    ax.plot(x[:, 0], gp_x.mu, color=color, **mean_kwargs, label='Mean')
    
    yy = jnp.linspace(*yrange, yres)[:, None]
    
    X, Y = jnp.meshgrid(x[:, 0], yy[:, 0])
    ax.pcolormesh(X, Y, (gp_shading(yy, gp_x.mu, gp_x.std)), norm=colors.PowerNorm(gamma=gamma),**std_kwargs)
    ax.plot(x[:, 0], gp_x.mu-2*gp_x.std, color=rgb.tue_blue, lw=.25)
    ax.plot(x[:, 0], gp_x.mu+2*gp_x.std, color=rgb.tue_blue, lw=.25)
    
    if num_samples>0:
      ax.plot(
        x[:, 0],
        gp_x.sample(rng_key, num_samples).T,
        color=rgb.tue_red,
        alpha=0.8,
        lw=.2,
      )
      
  def __add__(self, other):
    return GaussianProcessSum(self, other)
      

@dataclasses.dataclass
class GaussianProcessSum(GaussianProcess):
  """Sum of Gaussian Processes"""
  
  def __init__(self, *gps):
    self.gps = gps
    super().__init__(self._mean, self._covariance)
    
  def _mean(self, x):
    return sum(gp.m(x) for gp in self.gps)
  
  def _covariance(self, a, b):
    return sum(gp.k(a, b) for gp in self.gps)

@dataclasses.dataclass
class ConditionalGausianProcess(GaussianProcess):
  def __init__(self, prior: GaussianProcess, y: jnp.ndarray, X: jnp.ndarray, epsilon: Gaussian):
    self.prior = prior
    self.y = jnp.atleast_1d(y)
    self.X = jnp.atleast_2d(X)
    self.epsilon = epsilon
    super().__init__(self._mean, self._covariance)
  
  @functools.cached_property
  def predictive_covariance(self):
    return self.prior.k(self.X[:, None, :], self.X[None, :, :]) + self.epsilon.Sigma
  
  @functools.cached_property
  def predictive_covariance_cho(self):
    return jax.scipy.linalg.cho_factor(self.predictive_covariance, lower=True)
  
  @functools.cached_property
  def representer_weights(self):
    return jax.scipy.linalg.cho_solve(self.predictive_covariance_cho, self.y - self.prior(self.X).mu - self.epsilon.mu)
  
  def _mean(self, x):
    x = jnp.asarray(x)
    
    return (
      self.prior(x).mu
      + self.prior.k(x[..., None, :], self.X[None, :, :]) @ self.representer_weights
    )
    
  @functools.partial(jnp.vectorize, signature='(d),(d)->()', excluded={0})
  def _covariance(self, a, b):
    return (
      self.prior.k(a, b)
      - self.prior.k(a, self.X) @ jax.scipy.linalg.cho_solve(
        self.predictive_covariance_cho, self.prior.k(self.X, b)
        )
    )
  
  def _m_proj(self, x, projection, projection_mean):
    x = jnp.asarray(x)
    if projection_mean is None:
      projection_mean = self.prior.m
    
    return (
      projection_mean(x)
      + projection(x[..., None, :], self.X[None, :, :]) @ self.representer_weights
    )
  
  @functools.partial(jnp.vectorize, signature="(d),(d)->()", excluded={0,3})
  def _cov_proj(self, a, b, projection):
    return projection(a, b) - projection(a, self.X) @ jax.scipy.linalg.cho_solve(
      self.predictive_covariance_cho,
      projection(self.X, b)
    )
    
  def project(self, k_proj, m_proj=None):
    return GaussianProcess(
      lambda x: self._m_proj(x, k_proj, m_proj),
      lambda x0, x1: self._cov_proj(x0, x1, k_proj)
    )
    
def gp_shading(yy, mu, std):
  shading = jnp.exp(-0.5 * (yy - mu) ** 2 / std ** 2) / (std * jnp.sqrt(2 * jnp.pi))
  whitened = (shading-jnp.min(shading))/(jnp.max(shading)-jnp.min(shading))
  return whitened