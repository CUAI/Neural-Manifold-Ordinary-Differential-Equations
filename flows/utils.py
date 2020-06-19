import math
import torch
import torch.nn as nn
import os


EPS = {torch.float32: 1e-4, torch.float64: 1e-8}


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + EPS[x.dtype], 1 - EPS[x.dtype])
        ctx.save_for_backward(x)
        res = (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        positive_case = x + torch.sqrt(1 + x.pow(2))
        negative_case = 1 / (torch.sqrt(1 + x.pow(2)) - x)
        return torch.where(x > 0, positive_case, negative_case).log()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Acosh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        x = torch.clamp(x, min=1+EPS[x.dtype])
        z = torch.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z.data.clamp(min=EPS[z.dtype])
        z = g / z
        return z, None


artanh = Artanh.apply


arsinh = Arsinh.apply


arcosh = Acosh.apply

cosh_bounds = {torch.float32: 85, torch.float64: 700}
sinh_bounds = {torch.float32: 85, torch.float64: 500}


def cosh(x):
    x.data.clamp_(max=cosh_bounds[x.dtype])
    return torch.cosh(x)


def sinh(x):
    x.data.clamp_(max=sinh_bounds[x.dtype])
    return torch.sinh(x)


def tanh(x):
    return x.tanh()


def sqrt(x):
    return torch.sqrt(x).clamp_min(EPS[x.dtype])


class Sinhdiv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = sinh(x) / x
        y_stable = torch.ones_like(x)
        ctx.save_for_backward(x)
        return torch.where(x < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        y = (x * cosh(x) - sinh(x)) / x.pow(2)
        y_stable = torch.zeros_like(x)
        return torch.where(x < EPS[x.dtype], y_stable, y) * g


sinhdiv = Sinhdiv.apply


class Divsinh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = x / sinh(x)
        y_stable = torch.ones_like(x)
        ctx.save_for_backward(x)
        return torch.where(x < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        y = (1 - x * cosh(x) / sinh(x)) / sinh(x)
        y_stable = torch.zeros_like(x)
        return torch.where(x < EPS[x.dtype], y_stable, y) * g


divsinh = Divsinh.apply


class Sindiv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = torch.sin(x) / x
        y_stable = torch.ones_like(x)
        ctx.save_for_backward(x)
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        y = (x * torch.cos(x) - torch.sin(x)) / x.pow(2)
        y_stable = torch.zeros_like(x)
        # if torch.isnan(torch.where(x > 1 - EPS[x.dtype], y_stable, y)).any():
        #     raise ValueError("1")
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * g


sindiv = Sindiv.apply


class Divsin(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = x / torch.sin(x)
        y_stable = torch.ones_like(x)
        ctx.save_for_backward(x)
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        y = (1 - x * torch.cos(x) / torch.sin(x)) / torch.sin(x)
        y_stable = torch.zeros_like(x)
        # if torch.isnan(torch.where(x > 1 - EPS[x.dtype], y_stable, y)).any():
        #     raise ValueError("2")
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * g


divsin = Divsin.apply


class LeakyClamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, min, max):
        ctx.save_for_backward(x.ge(min) * x.le(max))
        return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * EPS[grad_output.dtype], None, None


def clamp(x, min=float("-inf"), max=float("+inf")):
    return LeakyClamp.apply(x, min, max)


def logsinh(x):
    # torch.log(sinh(x))
    # return x + torch.log(clamp(1. - torch.exp(-2. * x), min=eps)) - ln_2
    x_exp = x.unsqueeze(dim=-1)
    signs = torch.cat((torch.ones_like(x_exp), -torch.ones_like(x_exp)), dim=-1)
    value = torch.cat((torch.zeros_like(x_exp), -2. * x_exp), dim=-1)
    return x + logsumexp_signs(value, dim=-1, signs=signs) - math.log(2)


def logsumexp_signs(value, dim=0, keepdim=False, signs=None):
    if signs is None:
        signs = torch.ones_like(value)
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(clamp(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim), min=EPS[value.dtype]))


class MultiInputSequential(nn.Sequential):
    def forward(self, *input):
        multi_inp = False
        if len(input) > 1:
            multi_inp = True
            _, edge_index = input[0], input[1]

        for module in self._modules.values():
            if multi_inp:
                if hasattr(module, 'weight'):
                    input = [module(*input)]
                else:
                    # Only pass in the features to the Non-linearity
                    input = [module(input[0]), edge_index]
            else:
                input = [module(*input)]
        return input[0]

def check_mkdir(path, increment=False):
    r"""Only creates a directory if it does not exist already.  Emits an
    warning if it exists. When 'increment' is true, it creates a directory
    nonetheless by incrementing an integer at its end.
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        if increment:
            trailing_int = 0
            while os.path.isdir(path):
                basename = os.path.basename(path)
                split = basename.split('_')
                if split[-1].isdigit():
                    basename = '_'.join(split[:-1])
                path = os.path.join(
                        os.path.dirname(path),
                        basename + '_{}'.format(trailing_int))
                trailing_int += 1
            os.makedirs(path)
            print('Created the directory (%s) instead', path)
        else:
            print('The given path already exists (%s)', path)

    return path

