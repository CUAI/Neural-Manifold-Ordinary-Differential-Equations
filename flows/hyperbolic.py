import torch

from flows.manifold import Manifold
from flows.utils import EPS, cosh, sinh, tanh, arcosh, arsinh, artanh, sinhdiv, divsinh, logsinh


class Lorentz(Manifold):

    def __init__(self):
        super(Lorentz, self).__init__()

    @staticmethod
    def ldot(u, v, keepdim=False, dim=-1):
        m = u * v
        if keepdim:
            ret = torch.sum(m, dim=dim, keepdim=True) - 2 * m[..., 0:1]
        else:
            ret = torch.sum(m, dim=dim, keepdim=False) - 2 * m[..., 0]
        return ret

    def _ldot(self, u, v, keepdim=False, dim=-1):
        try:
            m = u * v
        except:
            m = torch.mm(u,v)
        if keepdim:
            ret = torch.sum(m, dim=dim, keepdim=True) - 2 * m[..., 0:1]
        else:
            ret = torch.sum(m, dim=dim, keepdim=False) - 2 * m[..., 0]
        return ret

    def sh_to_dim(self, sh):
        if hasattr(sh, '__iter__'):
            return sh[-1] - 1
        else:
            return sh - 1

    def dim_to_sh(self, dim):
        if hasattr(dim, '__iter__'):
            return dim[-1] + 1
        else:
            return dim + 1

    def zero(self, *shape):
        x = torch.zeros(*shape)
        x[..., 0] = 1
        return x

    def zero_vec(self, *shape):
        return torch.zeros(*shape)

    def zero_like(self, x):
        y = torch.zeros_like(x)
        y[..., 0] = 1
        return y

    def zero_vec_like(self, x):
        return torch.zeros_like(x)

    def inner(self, x, u, v, keepdim=False):
        return self._ldot(u, v, keepdim=keepdim)

    def proju(self, x, u, inplace=False):
        if not inplace:
            u = u.clone()
        return u.addcmul(self._ldot(x, u, keepdim=True).expand_as(u), x)

    def projx(self, x):
        x = x.clone()
        x.data[..., 0] = (1 + x[..., 1:].pow(2).sum(dim=-1)).sqrt()
        return x

    def egrad2rgrad(self, x, u):
        z = torch.eye(u.size(dim), dtype=x.dtype)
        z[0][0] = -1.
        u = u @ z
        u = self.proju(x, u)
        return u

    def exp(self, x, u):
        un = self._ldot(u, u, keepdim=True)
        un = un.clamp(min=EPS[x.dtype]).sqrt()
        return x.clone().mul(cosh(un)) + sinhdiv(un) * u

    def log(self, x, y):
        xy = self._ldot(x, y, keepdim=True)
        num = arcosh(-xy)
        u = divsinh(num) * (y + xy * x)
        return self.proju(x, u)

    def dist(self, x, y, squared=False, keepdim=False):
        d = -self._ldot(x, y)
        d.data.clamp(min=1)
        dist = arcosh(d)
        dist.data.clamp(min=EPS[x.dtype])
        return dist.pow(2) if squared else dist

    def transp(self, x, y, u):
        xy = self._ldot(x, y, keepdim=True).expand_as(u)
        uy = self._ldot(u, y, keepdim=True).expand_as(u)
        return u + uy / (1 - xy) * (x + y)

    def rand(self, *shape, ir=1e-2):
        x = torch.empty(*shape).uniform_(-ir, ir)
        return self.projx(x)

    def randvec(self, x, norm=1):
        vs = torch.rand(x.shape)
        vs[..., -1].zero_()
        vs.div_(vs.norm(dim=-1).unsqueeze(-1))
        zero = self.zero(x.shape)
        us = self.transp(zero, x, vs)
        return us

    def __str__(self):
        return 'Hyperboloid'

    def squeeze_tangent(self, x):
        return x[..., 1:]

    def unsqueeze_tangent(self, x):
        return torch.cat((torch.zeros_like(x[..., 0]).unsqueeze(-1), x), dim=-1)

    def logdetexp(self, x, u):
        val = self.norm(x, u)
        return (u.shape[-1] - 2) * sinhdiv(val).log()

    def to_poincare(self, x):
        return torch.div(x[..., 1:], (1+x[...,0]).reshape(x.shape[0], 1))
