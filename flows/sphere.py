import numpy as np
import torch
import flows

from flows.manifold import Manifold
from flows.utils import EPS, sindiv, divsin


class FirstJacobianScalar(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = x * torch.acos(x) / (1 - x.pow(2)).pow(1.5) - 1 / (1 - x.pow(2))
        y_limit = -torch.ones_like(x) / 3
        ctx.save_for_backward(x)
        return torch.where(x > 1 - EPS[x.dtype], y_limit, y)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        y = (-3 * (1 - x.pow(2)).sqrt() * x + 2 * x.pow(2) * torch.acos(x) + torch.acos(x)) / (1 - x.pow(2)).pow(2.5)
        y_limit = torch.ones_like(x) * 4/15
        return torch.where(x > 1 - EPS[x.dtype], y_limit, y) * g


firstjacscalar = FirstJacobianScalar.apply


class Sphere(Manifold):

    def __init__(self):
        super(Sphere, self).__init__()

    def zero(self, *shape, out=None):
        x = torch.zeros(*shape, out=out)
        x[..., 0] = -1
        return x

    def zero_vec(self, *shape, out=None):
        return torch.zeros(*shape, out=out)

    def zero_like(self, x):
        y = torch.zeros_like(x)
        y[..., 0] = -1
        return y

    def zero_vec_like(self, x):
        return torch.zeros_like(x)

    def inner(self, x, u, v, keepdim=False):
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def proju(self, x, u, inplace=False):
        return u.addcmul(-self.inner(None, x, u, keepdim=True), x)

    def projx(self, x, inplace=False):
        return x.div(self.norm(None, x, keepdim=True))

    def exp(self, x, u):
        norm_u = u.norm(dim=-1, keepdim=True)
        return x * torch.cos(norm_u) + u * sindiv(norm_u)

    def retr(self, x, u):
        return self.projx(x + u)

    def log(self, x, y):
        xy = (x * y).sum(dim=-1, keepdim=True)
        xy.data.clamp_(min=-1 + 1e-6, max=1 - 1e-6)
        val = torch.acos(xy)
        return divsin(val) * (y - xy * x)


    def jacoblog(self, x, y):
        z = (x * y).sum(dim=-1, keepdim=True)
        z.data.clamp_(min=-1 + 1e-4, max=1 - 1e-4)

        firstterm = firstjacscalar(z.unsqueeze(-1)) * (y - z * x).unsqueeze(-1) * x.unsqueeze(-2)
        secondterm = divsin(torch.acos(z).unsqueeze(-1)) * (torch.eye(x.shape[-1]).to(x).unsqueeze(0) - x.unsqueeze(-1) * x.unsqueeze(-2))
        return firstterm + secondterm


    def dist(self, x, y, squared=False, keepdim=False):
        inner = self.inner(None, x, y, keepdim=keepdim)
        inner.data.clamp_(min=-1 + EPS[x.dtype]**2, max=1 - EPS[x.dtype]**2)
        sq_dist = torch.acos(inner)
        sq_dist.data.clamp_(min=EPS[x.dtype])

        return sq_dist.pow(2) if squared else sq_dist

    def rand(self, *shape, out=None, ir=1e-2):
        x = self.zero(*shape, out=out)
        u = self.randvec(x, norm=ir)
        return self.retr(x, u)

    def rand_uniform(self, *shape, out=None):
        return self.projx(
                torch.randn(*shape, out=out), inplace=True)

    def rand_ball(self, *shape, out=None):
        xs_unif = self.rand_uniform(*shape, out=out)
        rs = torch.rand(*shape[0]).pow_(1 / (self.dim + 1))
        # rs = rs.reshape(*shape, *((1, ) * len(self.shape)))
        xs_ball = xs_unif.mul_(rs)
        return xs_ball

    def randvec(self, x, norm=1):
        u = torch.randn(x.shape, out=torch.empty_like(x))
        u = self.proju(x, u, inplace=True)  # "transport" ``u`` to ``x``
        u.div_(u.norm(dim=-1, keepdim=True)).mul_(norm)  # normalize
        return u

    def transp(self, x, y, u):
        yu = torch.sum(y * u, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        return u - yu/(1 + xy) * (x + y)

    def __str__(self):
        return "Sphere"
    
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

    def squeeze_tangent(self, x):
        return x[..., 1:]

    def unsqueeze_tangent(self, x):
        return torch.cat((torch.zeros_like(x[..., :1]), x), dim=-1)  

    def logdetexp(self, x, u):
        norm_u = u.norm(dim=-1)
        val = torch.abs(sindiv(norm_u)).log()
        return (u.shape[-1]-2) * val
