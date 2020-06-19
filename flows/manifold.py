import abc
import torch
import torch.autograd.functional as AF
import numpy as np

from flows.utils import EPS


class Manifold(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def zero(self, *shape):
        pass

    @abc.abstractmethod
    def zero_like(self, x):
        pass

    @abc.abstractmethod
    def zero_vec(self, *shape):
        pass

    @abc.abstractmethod
    def zero_vec_like(self, x):
        pass

    @abc.abstractmethod
    def inner(self, x, u, v, keepdim=False):
        pass

    def norm(self, x, u, squared=False, keepdim=False):
        norm_sq = self.inner(x, u, u, keepdim)
        norm_sq.data.clamp_(EPS[u.dtype])
        return norm_sq if squared else norm_sq.sqrt()

    @abc.abstractmethod
    def proju(self, x, u):
        pass

    def proju0(self, u):
        return self.proju(self.zero_like(u), u)

    @abc.abstractmethod
    def projx(self, x):
        pass

    def egrad2rgrad(self, x, u):
        return self.proju(x, u)

    @abc.abstractmethod
    def exp(self, x, u):
        pass

    def exp0(self, u):
        return self.exp(self.zero_like(u), u)

    @abc.abstractmethod
    def log(self, x, y):
        pass

    def log0(self, y):
        return self.log(self.zero_like(y), y)
        
    def dist(self, x, y, squared=False, keepdim=False):
        return self.norm(x, self.log(x, y), squared, keepdim)

    def pdist(self, x, squared=False):
        assert x.ndim == 2
        n = x.shape[0]
        m = torch.triu_indices(n, n, 1, device=x.device)
        return self.dist(x[m[0]], x[m[1]], squared=squared, keepdim=False)

    def transp(self, x, y, u):
        return self.proju(y, u)

    def transpfrom0(self, x, u):
        return self.transp(self.zero_like(x), x, u)
    
    def transpto0(self, x, u):
        return self.transp(x, self.zero_like(x), u)

    def mobius_addition(self, x, y):
        return self.exp(x, self.transp(self.zero_like(x), x, self.log0(y)))

    @abc.abstractmethod
    def sh_to_dim(self, shape):
        pass

    @abc.abstractmethod
    def dim_to_sh(self, dim):
        pass

    @abc.abstractmethod
    def squeeze_tangent(self, x):
        pass

    @abc.abstractmethod
    def unsqueeze_tangent(self, x):
        pass

    @abc.abstractmethod
    def rand(self, *shape):
        pass

    @abc.abstractmethod
    def randvec(self, x, norm=1):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    def logdetexp(self, x, u):
        #very expensive rip
        if len(u.shape) == 1:
            return torch.det(AF.jacobian(lambda v: self.exp(x, v), u))
        else:
            jacobians = [AF.jacobian(lambda v: self.exp(x[i], v), u[i]) for
                    i in range(u.shape[0])]
            return torch.det(torch.stack(jacobians))

    def logdetlog(self, x, y):
        return -self.logdetexp(x, self.log(x, y))

    def logdetexp0(self, u):
        return self.logdetexp(self.zero_like(u), u)
    
    def logdetlog0(self, y):
        return self.logdetlog(self.zero_like(y), y)