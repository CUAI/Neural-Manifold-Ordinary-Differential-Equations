import argparse
import torch
from torch import optim
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
import os
import glob

from flows.mcnf import SCNF, HCNF
from flows.hyperbolic import Lorentz
from flows.sphere import Sphere
from flows.utils import check_mkdir
from distributions.wnormal import WrappedNormal
from distributions.vmf import VonMisesFisher
from test_densities.density_hyp import data_gen_hyp, plot_poincare_density
from test_densities.density_sphere import data_gen_sphere
from test_densities.density_hyp import plot_distr as plot_distr_hyp
from test_densities.density_sphere import plot_distr as plot_distr_sphere

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default='1e-3')
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--dataset', type=str, default='1wrapped', help='for hyperboloid: 1wrapped | 5gaussians | bigcheckerboard | mult_wrapped, for the sphere: 1wrapped | 4wrapped | bigcheckerboard')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--num_drops', default=2, type=int, help='number of times to drop the learning rate')
parser.add_argument('--flow_hidden_size', type=int, default=32, \
                    help='Hidden layer size for flows.')
parser.add_argument('--save', action='store_true', default=False, help='Save a visualization of the learned density')
parser.add_argument('--conc', type=float, default=1,
                        help='Concentration of vMF')
parser.add_argument('--dev', type=str, default='cuda')
parser.add_argument('--M', type=str, default='Hyperboloid',
                        choices=['Hyperboloid', 'Sphere'])
parser.add_argument('--contsave', action='store_true', default=False,
                        help='Continuously save intermediate flow visualization in contsave/')
parser.add_argument('--save_freq', type=int, default=5,
                        help='frequency of continuous saving of intermediate flows')

args = parser.parse_args()

if args.dev == 'cuda':
    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    args.dev = torch.device(args.dev)

if args.M == 'Hyperboloid':
    M = Lorentz()
    loc = M.zero((3,)).to(args.dev)
    scale = torch.ones(1,2).to(args.dev)
    base_distr = WrappedNormal(M, loc, scale)
    make_model = HCNF
    plot_distr = plot_distr_hyp
    data_gen = data_gen_hyp

elif args.M == 'Sphere':
    M = Sphere()
    loc = M.zero((3,)).to(args.dev)
    scale = args.conc*torch.ones(1).to(args.dev)
    base_distr = VonMisesFisher(loc, scale)
    make_model = SCNF
    plot_distr = plot_distr_sphere
    data_gen = data_gen_sphere




def compute_loss(args, model, x):

    # transform to z
    z, delta_logp = model(x)

    logpz = base_distr.log_prob(z)
    logpx = logpz + delta_logp
    loss = -torch.mean(logpx)

    return loss


def main(args):

    input_dim, z_dim = 2, 3
    model = make_model(input_dim, args.flow_hidden_size, z_dim).to(args.dev)


    print(vars(args))

    ### Start Training ###
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batch_size = args.batch_size

    # number of times to drop learning rate
    num_drops = args.num_drops
    lr_milestones = [j*args.epochs//(num_drops+1) for j in range(1,num_drops+1)]
    scheduler = optim.lr_scheduler.MultiStepLR(opt, lr_milestones, gamma=.1)

    if args.contsave:
        check_mkdir(f'contsave/{args.dataset}_{args.M}/')
        files = glob.glob(f'contsave/{args.dataset}_{args.M}/*')
        for f in files:
            os.remove(f)

    for epoch in range(0, args.epochs):
        samples = data_gen(args.dataset, batch_size).to(args.dev)

        opt.zero_grad()

        loss = compute_loss(args, model, samples)
        loss.backward()
        opt.step()
        scheduler.step()
        train_loss = loss.item()/batch_size

        if epoch % 2 == 0:
            print(f'Epoch: {epoch}, Loss: {train_loss}')

        if args.contsave and epoch % args.save_freq == 0:
            namestr = f'contsave/{args.dataset}_{args.M}/{epoch}'
            plot_distr(distr='model', model=model, device=args.dev, save_fig=args.save,
                    base_distr=base_distr, namestr=namestr)
            plt.close()

        
    # make grid, plot density evaluated on grid
    namestr = f'{args.dataset}_{args.M}'
    plot_distr(distr='model', model=model, device=args.dev, save_fig=args.save,
                base_distr=base_distr, namestr=namestr)
    
    return model

if __name__ == '__main__':
    model = main(args)
