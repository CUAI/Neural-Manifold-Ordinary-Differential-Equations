import torch
import torch.nn as nn
import torch.autograd.functional as AF
import numpy as np

from flows.hyperbolic import Lorentz
from flows.sphere import Sphere
from torchdiffeq import odeint_adjoint as odeint
from flows.utils import MultiInputSequential

sphere = Sphere()
hyperbolic = Lorentz()


def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def create_network(input_size, output_size, hidden_size, n_hidden):
    net = [nn.Linear(input_size, hidden_size)]
    for _ in range(n_hidden):
        net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
    net += [nn.Tanh(), nn.Linear(hidden_size, output_size)]
    return MultiInputSequential(*net)


class TimeNetwork(nn.Module):
    def __init__(self, func):
        super(TimeNetwork, self).__init__()
        self.func = func

    def forward(self, t, x):
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        t_p = t.expand(x.shape[:-1] + (1,))
        return self.func(torch.cat((x, t_p), -1))


class ODEfunc(nn.Module):

    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()

        self.diffeq = diffeq
        
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        if not torch.is_tensor(t):
            t = torch.tensor(t).to(y)
        batchsize = y.shape[0]
        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            dy = self.diffeq(t, y)
            divergence = divergence_bf(dy, y).unsqueeze(-1)

        return tuple([dy, -divergence])


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class SphereProj(nn.Module):
    def __init__(self, func, loc):
        super(SphereProj, self).__init__()
        self.base_func = func
        self.man = sphere
        self.loc = loc.detach()

    def forward(self, t, x):
        """
        x is assumed to be an input on the tangent space of self.loc
        """
        y = self.man.exp(self.loc, x)
        val = self.man.jacoblog(self.loc, y) @ self.base_func(t, y).unsqueeze(-1)
        val = val.squeeze()
        return val


class AmbientProjNN(nn.Module):
    def __init__(self, func):
        super(AmbientProjNN, self).__init__()
        self.func = func
        self.man = sphere

    def forward(self, t, x):
        x = self.man.proju(x, self.func(t, x))
        return x


class SCNF(nn.Module):
    def __init__(self, input_size, flow_hidden_size, n_hidden):
        super(SCNF, self).__init__()

        self.solver = 'rk4'
        self.atol = 1e-2
        self.rtol = 1e-2
        self.solver_options = {'step_size': 1/16}
        self.man = sphere

        amb_dim = self.man.dim_to_sh(input_size)
        self.func = AmbientProjNN(TimeNetwork(create_network(amb_dim + 1, amb_dim, flow_hidden_size, n_hidden)))

    def forward(self, z, charts=4):
        return self._forward(z, charts=charts)

    def inverse(self, z, charts=4):
        return self._forward(z, reverse=True, charts=charts)

    def _forward(self, z, reverse=False, charts=4):
        integration_times = torch.tensor(
            [[i/charts, (i + 1)/charts] for i in range(charts)]
        )
        if reverse:
            #flip each time steps [s_t, e_t]
            integration_times = _flip(integration_times, -1)
            #reorder time steps from 0 -> n to give n -> 0
            integration_times = _flip(integration_times, 0)

        # initial values
        loc = z.detach()
        tangval = self.man.log(loc, z)

        logpz_t = 0

        for time in integration_times:
            chartproj = SphereProj(self.func, loc)
            chartfunc = ODEfunc(chartproj)

            logpz_t -= self.man.logdetexp(loc, tangval)

            # integrate as a tangent space operation
            state_t = odeint(
                    chartfunc,
                    (tangval, torch.zeros(tangval.shape[0], 1).to(tangval)),
                    time.to(z),
                    atol=self.atol,
                    rtol=self.rtol,
                    method=self.solver,
                    options=self.solver_options
                )

            # extract information
            state_t = tuple(s[1] for s in state_t)
            y_t, logpy_t = state_t[:2]
            y_t = self.man.proju(loc, y_t)


            # log p updates
            logpz_t -= logpy_t.squeeze()
            logpz_t += self.man.logdetexp(loc, y_t)

            # set up next iteration values
            z_n = self.man.exp(loc, y_t)
            loc = z_n
            tangval = self.man.log(loc, z_n)

        return z_n, logpz_t
        
    def get_regularization_states(self):
        return None

    def num_evals(self):
        return self.odefunc._num_evals.item()


class HCNF(nn.Module):
    def __init__(self, input_size, flow_hidden_size, n_hidden):
        super(HCNF, self).__init__()

        self.odefunc = ODEfunc(TimeNetwork(create_network(input_size + 1, input_size, flow_hidden_size, n_hidden)))
        self.solver = 'dopri5'
        self.atol = 1e-3
        self.rtol = 1e-3
        self.solver_options = {}
        self.man = hyperbolic

    def forward(self, z):
        return self._forward(z)

    def inverse(self, z):
        return self._forward(z, reverse=True)

    def _forward(self, z, reverse=False):

        integration_times = torch.tensor([0.0, 1.0]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        y_orig = self.man.log0(z)
        y = self.man.squeeze_tangent(y_orig)

        state_t = odeint(
                self.odefunc,
                (y, torch.zeros(y.shape[0], 1).to(y)),
                integration_times.to(z),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                options=self.solver_options
            )
            
        state_t = tuple(s[1] for s in state_t)

        y_t, logpy_t = state_t[:2]
        y_t = self.man.unsqueeze_tangent(y_t)

        z_t = self.man.exp0(y_t)
        if not reverse:
            logpz_t = -logpy_t.squeeze() - self.man.logdetexp(self.man.zero_like(y_t), y_t) + self.man.logdetexp(self.man.zero_like(y_orig), y_orig)
        else:
            logpz_t = -logpy_t.squeeze() + self.man.logdetexp(self.man.zero_like(y_t), y_t) - self.man.logdetexp(self.man.zero_like(y_orig), y_orig)

        return z_t, logpz_t
        
    def get_regularization_states(self):
        return None

    def num_evals(self):
        return self.odefunc._num_evals.item()
