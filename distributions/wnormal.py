import torch
import torch.distributions

class WrappedNormal(torch.distributions.Distribution):
    arg_constraints = {
                'loc': torch.distributions.constraints.real_vector,
                'scale': torch.distributions.constraints.positive
                }

    support = torch.distributions.constraints.real
    has_rsample = True

    def __init__(self, manifold, loc, scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifold = manifold
        self.loc = loc
        self.scale = scale
        self.dev = self.loc.device
        self.normal = torch.distributions.Normal(self.manifold.squeeze_tangent(
            self.manifold.zero_vec_like(loc)), scale)

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    def rsample(self, shape=torch.Size()):
        # v ~ N(0, \Sigma)
        v = self.normal.rsample(shape)
        # u = PT_{mu_0 -> mu}([0, v_tilde])
        # z = exp_{mu}(u)
        u = self.manifold.transp(self.manifold.zero_like(self.loc), self.loc, 
                                self.manifold.unsqueeze_tangent(v))
        z = self.manifold.exp(self.loc, u)
        return z

    def log_prob(self, z):
        # log(z) = log p(v) - log det [(\partial / \partial v) proj_{\mu}(v)]
        u = self.manifold.log(self.loc, z)
        v = self.manifold.transp(self.loc, self.manifold.zero_like(self.loc), u)
        v = self.manifold.squeeze_tangent(v)
        n_logprob = self.normal.log_prob(v).sum(dim=-1)
        logdet = self.manifold.logdetexp(self.loc, u)
        assert n_logprob.shape == logdet.shape
        log_prob = n_logprob - logdet
        return log_prob

    def rsample_log_prob(self, shape=torch.Size()):
        z = self.rsample(shape)
        return z, self.log_prob(z)
