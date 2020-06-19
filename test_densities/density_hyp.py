import torch
import math
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
import matplotlib.pyplot as plt
import matplotlib

from flows.hyperbolic import Lorentz
from distributions.wnormal import WrappedNormal

M = Lorentz()

## Data fetching

def data_gen_hyp(dataset, n_samples=100):
    """Samples from the distribution given by dataset"""
    z = torch.randn(n_samples, 2)

    if dataset == '1wrapped':
        mu = torch.Tensor([-1., 1.]).unsqueeze(0)
        std_v = .75
        std_1 = torch.Tensor([[std_v], [std_v]]).T
        radius = torch.ones(1)
        mu_h = M.exp0(M.unsqueeze_tangent(mu))
        distr = WrappedNormal(M, mu_h, std_1)
        samples = distr.rsample((n_samples,)).squeeze()
        return samples
    
    elif dataset == '5gaussians':
        scale = 3
        z = z/2
        centers = [(1,0), (-1,0), (0,1), (0,-1), (0,0)]
        centers = torch.tensor([(scale * x, scale * y) for x,y in centers])
        samples = z + centers[torch.randint(len(centers), size=(n_samples,))]

    elif dataset == 'bigcheckerboard':
        s = 1.5 # side length
        offsets = [(0,0), (0, -2*s), (s,s), (s, -s), (-s, s), (-s, -s), (-2*s, 0), (-2*s, -2*s)]
        offsets = torch.tensor([o for o in offsets])

        # (x,y) ~ uniform([0,s] \times [0,s])
        x1 = torch.rand(n_samples) * s
        x2 = torch.rand(n_samples) * s

        samples = torch.stack([x1, x2], dim=1)
        samples = samples + offsets[torch.randint(len(offsets), size=(n_samples,))]
    
    elif dataset == 'mult_wrapped':
        s = 1.3
        centers = [torch.tensor([[0., s, s]]), torch.tensor([[0, -s, -s]]), 
                   torch.tensor([[0., -s, s]]), torch.tensor([[0, s, -s]])]
        centers = [M.projx(center) for center in centers]
        n = n_samples//len(centers)
        var1 = .3
        var2 = 1.5
        scales = [torch.tensor([[var1, var2]]), torch.tensor([[var1, var2]]), 
                  torch.tensor([[var2, var1]]), torch.tensor([[var2, var1]]) ]

        distrs = []
        for i in range(len(centers)):
            loc = centers[i]
            scale = scales[i]
            distrs.append(WrappedNormal(M, loc, scale))
        samples = distrs[0].rsample((n,))
        for distr in distrs[1:]:
            samples = torch.cat([samples, distr.rsample((n,))], dim=0)
        samples = samples.squeeze()
        return samples

    # transform to Lorentz
    threedim = M.unsqueeze_tangent(samples)
    samples = M.exp0(threedim)

    return samples

## Plotting utility

def plot_poincare_density(xy_poincare, prob, npts, cmap='magma', uniform=False):
    plt.figure()
    x = xy_poincare[:,0].cpu().numpy().reshape(npts, npts)
    y = xy_poincare[:,1].cpu().numpy().reshape(npts, npts)
    prob = prob.detach().cpu().numpy().reshape(npts, npts)
    if not uniform:
        plt.pcolormesh(x, y, prob, cmap=cmap)
    else: # uniform color
        colormap = plt.cm.get_cmap(cmap)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        prob[prob > 0] = .5
        plt.pcolormesh(x, y, prob, cmap=cmap, norm=norm)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('off')
    plt.axis('equal')

## Plot true and model code

def plot_distr(distr=None, res_npts=500, save_fig=True, model=None, device=None,
        base_distr=None, namestr='hyp_model'):
    on_mani, xy, log_detjac, twodim = make_grid_hyp(res_npts)

    if distr == '1wrapped':
        probs = true_1wrapped_probs(on_mani)
    elif distr == '5gaussians':
        probs = true_5gaussians_probs(on_mani, twodim)
    elif distr == 'bigcheckerboard':
        probs = true_bigcheckerboard_probs(on_mani, twodim)
    elif distr == 'mult_wrapped':
        probs = true_mult_wrapped_probs(on_mani)
    elif distr == 'model':
        probs = model_probs(model, on_mani, log_detjac, device, base_distr)

    plot_poincare_density(xy, probs, res_npts, uniform = True if distr == 'bigcheckerboard' else False)

    if save_fig:
      print(f'Saved to: {namestr}.png')
      plt.savefig(f'{namestr}.png')


def true_1wrapped_probs(on_mani):
    mu = torch.Tensor([-1., 1.]).unsqueeze(0)
    std_v = .75
    std_1 = torch.Tensor([[std_v], [std_v]]).T
    radius = torch.ones(1)
    mu_h = M.exp0(M.unsqueeze_tangent(mu))
    distr = WrappedNormal(M, mu_h, std_1)
    prob = torch.exp(distr.log_prob(on_mani))
    return prob


def true_5gaussians_probs(on_mani, twodim):
    scale = 3
    centers = [(1,0), (-1,0), (0,1), (0,-1), (0,0)]
    centers = torch.tensor([(scale * x, scale * y) for x,y in centers])

    prob = 0
    for c in centers:
        loc = torch.ones_like(twodim) * torch.tensor(c)
        distr = MultivariateNormal(loc, .25*torch.eye(2))
        prob += torch.exp(distr.log_prob(twodim))
    prob /= len(centers)
    return prob


def true_bigcheckerboard_probs(on_mani, twodim):
    s = 1.5 # side length

    def in_board(z,s):
        """Whether z is in the checkerboard of side length s"""
        if 0 <= z[0] < s or -2*s <= z[0] < -s:
            return 0 <= z[1] < s or -2*s <= z[1] < -s
        elif -2*s <= z[0] < 2*s:
            return s <= z[1] < 2*s or -s <= z[1] < 0
        else:
            return 0
    
    prob = torch.zeros(twodim.shape[0])
    for i in range(twodim.shape[0]):
        prob[i] = in_board(twodim[i,:],s)

    prob /= torch.sum(prob)
    return prob


def true_mult_wrapped_probs(on_mani): 
    s = 1.3
    centers = [torch.tensor([[0., s, s]]), torch.tensor([[0, -s, -s]]), 
               torch.tensor([[0., -s, s]]), torch.tensor([[0, s, -s]])]
    centers = [M.projx(center) for center in centers]
    n = on_mani.shape[0]
    var1 = .3
    var2 = 1.5
    scales = [torch.tensor([[var1, var2]]), torch.tensor([[var1, var2]]), 
              torch.tensor([[var2, var1]]), torch.tensor([[var2, var1]])]
    distrs = []
    for i in range(len(centers)):
        loc = centers[i].repeat(n,1)
        scale = scales[i].repeat(n,1)
        distrs.append(WrappedNormal(M, loc, scale))
    prob = torch.zeros(on_mani.shape[0])
    for distr in distrs:
        prob += torch.exp(distr.log_prob(on_mani))
    prob /= len(centers)
    return prob


def model_probs(model, on_mani, log_detjac, device, base_distr):
    if device:
        on_mani = on_mani.to(device)
        log_detjac = log_detjac.to(device)
    z, logprob = model(on_mani)

    val = base_distr.log_prob(z)
    val += logprob.detach()
    val += log_detjac
    probs = torch.exp(val)
    return probs


def make_grid_hyp(npts):
    bp = torch.linspace(-5, 5, npts)
    xx, yy = torch.meshgrid((bp, bp))
    twodim = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    threedim = M.unsqueeze_tangent(twodim)
    on_mani = M.exp0(threedim)
    xy = M.to_poincare(on_mani)
    dummy = -1
    log_detjac = -M.logdetexp(dummy, threedim)
    return on_mani, xy, log_detjac, twodim

