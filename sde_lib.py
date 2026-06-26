import abc
import torch
class SDE(abc.ABC):


  def __init__(self):

    super().__init__()

  @property
  @abc.abstractproperty
  def T(self):
    pass

  @abc.abstractmethod
  def drift(self, x, t):
    pass

  @abc.abstractmethod
  def diffusion(self, x, t):
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    pass

  @abc.abstractmethod
  def time_steps(self, num, device):
    pass

class VP(SDE):

  def __init__(self,T=5,delta=5e-3):
    super().__init__()
    self._T = T
    self.delta = delta

  def T(self):
    return self._T

  def scaling(self, t):
    return torch.exp(-t)

  def drift(self, x,t):
    return - x

  def diffusion(self, x,t):
    return (2)**.5

  def time_steps(self, n, device):
    from math import exp, log
    c = 1.6 * (exp(log(self.T()/self.delta)/n) - 1)
    t_steps = torch.zeros(n,device=device)
    t_steps[0] = self.delta
    exp_step = True
    for i in range(1,n):
      if exp_step:
        t_steps[i] = t_steps[i-1] + c * t_steps[i-1]
        if t_steps[i] >= 1:
          c = (self.T() - t_steps[i-1])/(n-i)
          t_steps[i] = t_steps[i-1] + c
          exp_step = False
      else:
        t_steps[i] = t_steps[i-1] + c

    t_steps[-1] = self.T()
    t_steps = self.T() - t_steps
    t_steps = torch.flip(t_steps,dims=(0,))
    return t_steps

  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float32, device=device)

def get_sde(config):
    if config.sde_type == 'vp':
        return VP(config.T, config.sampling_eps)
