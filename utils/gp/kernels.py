"""Kernels for Gaussian Processes"""
import dataclasses
import jax.numpy as jnp
import jax
import functools
import numpy as np

class Kernel:
  
    def __init__(self, kernel: callable):
        self.kernel = kernel
    
    def check_mercer(self, x1: jnp.ndarray, x2: jnp.ndarray):
        assert all(self(x1, x2) == self(x2, x1)), "Kernel is not symmetric"
        assert all(self(x1, x1) >= 0) & all(self(x2, x2)), "Kernel is not positive semi-definite"
    
    def __add__(self, other):
        return AddKernel(self, other)
      
    @functools.singledispatchmethod
    def __mul__(self, other: float):
        kernel = self.kernel
        updated_kernel = lambda x, y: other * kernel(x, y)
        
        return Kernel(updated_kernel)
    
    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        return self.kernel(x1, x2)
    
@Kernel.__mul__.register
def _mul_kernel(self, other: Kernel):

    return MulKernel(self, other)
  
class AddKernel(Kernel):
    def __init__(self, left:Kernel, right:Kernel):
      self.left = left
      self.right = right
      super().__init__(self._kernel)
      self._left_name = left.__str__()
      self._right_name = right.__str__()
      
    def _kernel(self, x1, x2):
      return self.left(x1, x2) + self.right(x1, x2)
    
    def __str__(self):
      return f"{self._left_name}+{self._right_name}".replace("()", "")
    
    def __repr__(self):
      return str(self)
    

class MulKernel(Kernel):
    def __init__(self, left: Kernel, right: Kernel):
        self.left = left
        self.right = right
        super().__init__(self._kernel)
    
    def _kernel(self, x1, x2):
        return self.left(x1, x2) * self.right(x1, x2)

@dataclasses.dataclass
class RBFKernel(Kernel):
  
    def __init__(self, length_scale: float, num_features: int=8, cmin: float=-8, cmax: float=8):
        self.length_scale = length_scale
        self.num_features = num_features
        self.cmin = cmin
        self.cmax = cmax
        super().__init__(self.rbf_kernel)

    def rbf_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        phi1 = self._gaussian_features(x1)
        phi2 = self._gaussian_features(x2)
        return (phi1 * phi2).sum(axis=-1)
      
    def _gaussian_features(self, x):
        c = jnp.linspace(self.cmin, self.cmax, self.num_features)
        phi = (self.cmax-self.cmin)*jnp.exp(-0.5 * (x - c) ** 2 / self.length_scale ** 2)/jnp.sqrt(self.num_features)
        return phi

@dataclasses.dataclass
class SEKernel(Kernel):

  def __init__(self, length_scale: float, theta: float=1.0):
    self.length_scale = length_scale
    self.theta = theta
    super().__init__(self._se_kernel)
    
  def _se_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return self.theta**2 * jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2, axis=-1) / self.length_scale ** 2)

@dataclasses.dataclass
class ParametricKernel(Kernel):
    
    def __init__(self, phi: callable):
        self.phi = phi
        super().__init__(self._parametric_kernel)
    
    def _parametric_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        phi_x1 = self.phi(x1)
        phi_x2 = self.phi(x2)
        return (phi_x1*phi_x2).sum(axis=-1)

@dataclasses.dataclass
class WeinerKernel(Kernel):
    
    def __init__(self, shift, theta: float=1.0):
      self.shift = shift
      self.theta = theta
      super().__init__(self._weiner_kernel)
      
    def _weiner_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
      return self.theta**2 * jnp.maximum(jnp.prod(jnp.minimum(x1, x2), axis=-1)-self.shift, 0)
  
@dataclasses.dataclass
class RQKernel(Kernel):
  def __init__(self, length_scale: float, alpha: float=1.0, theta: float=1.0):
    self.length_scale = length_scale
    self.alpha = alpha
    self.theta = theta
    super().__init__(self._rq_kernel)
  
  def _rq_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return self.theta**2 * (1 + jnp.sum((x1-x2)/self.length_scale, axis=-1)**2 / (2 * self.alpha)) ** (-self.alpha)

@dataclasses.dataclass
class LinearKernel(Kernel):
  
  def __init__(self, alpha: float=1.0, beta: float=1.0):
    self.alpha = alpha
    self.beta = beta
    super().__init__(self._linear_kernel)
  
  def __pow__(self, n:int):
    if n==0:
      return WhiteNoiseKernel()
    if n==1:
      return LinearKernel(self.alpha, self.beta)
    if n>1:
      k = LinearKernel(self.alpha, self.beta)
      for i in range(n-1):
        k = k*LinearKernel(self.alpha, self.beta)
      return k
    
  def _linear_kernel(self, x1, x2):
    return self.alpha**2 * jnp.sum(x1 * x2, axis=-1) + self.beta**2

@dataclasses.dataclass
class PolyKernel(LinearKernel):
  def __init__(self, alpha: float=1.0, beta: float=1.0, degree=3.0):
    self.degree = degree
    super().__init__(alpha, beta)
    self.kernel = self._poly_kernel
    
  def _poly_kernel(self, x1, x2):
    if self.degree == 0:
      return (LinearKernel(self.alpha, self.beta)**0).kernel(x1, x2)
    if self.degree >0:
      k = LinearKernel(self.alpha, self.beta)**0
      for i in range(1, self.degree+1):
        k = LinearKernel(self.alpha, self.beta)**i + k
      return k.kernel(x1, x2)

@dataclasses.dataclass
class PeriodicKernel(Kernel):
  def __init__(self, length_scale: float, period: float=1.0, theta: float=1.0):
    self.length_scale = length_scale
    self.period = period
    self.theta = theta
    super().__init__(self._periodic_kernel)
  
  def _periodic_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return self.theta**2 * jnp.exp(-2 * jnp.sin(jnp.pi * jnp.sum((x1-x2)/self.period, axis=-1))**2 / self.length_scale**2)

@dataclasses.dataclass
class WhiteNoiseKernel(Kernel):
  def __init__(self, sigma: float=1.0):
    self.sigma = sigma
    super().__init__(self._white_noise_kernel)
  
  def _white_noise_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return self.sigma**2 * jnp.all(x1 == x2, axis=-1)
  
@dataclasses.dataclass
class ConstantMean:
  mu: float
  
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return self.mu * jnp.ones(x.shape[0])