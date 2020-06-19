import torch
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

from flows.sphere import Sphere
from distributions.wnormal import WrappedNormal
from distributions.vmf import VonMisesFisher


## Utility methods for transformations

def xyz_to_spherical(xyz):
    # assume points live on hypersphere
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    lonlat = np.empty((xyz.shape[0], 2))
    phi = np.arctan2(y, x)
    phi[y<0] = phi[y<0] + 2*np.pi
    lonlat[:,0] = phi
    lonlat[:,1] = np.arctan2(np.sqrt(x**2 + y**2), z)
    return lonlat

def spherical_to_xyz(lonlat):
    # lonlat[:,0] is azimuth phi in [0,2pi]
    # lonlat[:,1] is inclination theta in [0,pi]

    phi = lonlat[:,0]
    theta = lonlat[:,1]
    x = torch.sin(theta)*torch.cos(phi)
    y = torch.sin(theta)*torch.sin(phi)
    z = torch.cos(theta)

    xyz = torch.stack([x,y,z], dim=1)
    detjac = -torch.sin(theta)
    return xyz, detjac


def plot_sphere_density(lonlat, probs, npts, uniform=False):
    # lon in [0,2pi], lat in [0,pi]
    fig = plt.figure(figsize=(3,2), dpi=200)
    proj = ccrs.Mollweide()
    ax = fig.add_subplot(111, projection=proj)
    lon, lat = lonlat[:,0], lonlat[:,1]
    lon = lon.cpu().numpy().reshape(npts, npts)
    lat = lat.cpu().numpy().reshape(npts, npts)
    lon -= np.pi
    lat -= np.pi/2
    probs = probs.cpu().numpy().reshape(npts,npts)
    if not uniform:
        ax.pcolormesh(lon*180/np.pi, lat*180/np.pi, probs,
                         transform=ccrs.PlateCarree(), cmap='magma')
    else: # uniform color
        colormap = plt.cm.get_cmap('magma')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        probs[probs > 0] = .5
        ax.pcolormesh(lon*180/np.pi, lat*180/np.pi, probs,
                         transform=ccrs.PlateCarree(), cmap='magma',
                        norm=norm)

    plt.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_global()


## Data fetching

def data_gen_sphere(dataset, n_samples):
    if dataset == '1wrapped':
        center = -torch.ones(3)
        center = Sphere().projx(center)
        loc = center.repeat(n_samples, 1)
        scale = torch.ones(n_samples, 2)*torch.tensor([.3,.3])
        distr = WrappedNormal(Sphere(), loc, scale)
        samples = distr.rsample()

    elif dataset == '4wrapped':
        one = torch.ones(3)
        oned = torch.ones(3)
        oned[2] = -1
        centers = [one, -one, oned, -oned]
        centers = [Sphere().projx(center) for center in centers]
        n = n_samples//len(centers)
        scales = [torch.ones(n,2)*torch.tensor([.3,.3]) for _ in range(len(centers))]
        distrs = []
        for i in range(len(centers)):
            loc = centers[i].repeat(n,1)
            distrs.append(WrappedNormal(Sphere(), loc, scales[i]))
        samples = distrs[0].rsample()
        for i in range(1, len(distrs)):
            samples = torch.cat([samples, distrs[i].rsample()], dim=0)

    elif dataset == 'bigcheckerboard':
        s = np.pi/2-.2 # long side length
        offsets = [(0,0), (s, s/2), (s, -s/2), (0, -s), (-s, s/2), (-s, -s/2), (-2*s, 0), (-2*s, -s)]
        offsets = torch.tensor([o for o in offsets])

        # (x,y) ~ uniform([pi,pi + s] times [pi/2, pi/2 + s/2])
        x1 = torch.rand(n_samples) * s + np.pi
        x2 = torch.rand(n_samples) * s/2 + np.pi/2

        samples = torch.stack([x1, x2], dim=1)
        off = offsets[torch.randint(len(offsets), size=(n_samples,))]

        samples += off

        samples, _ = spherical_to_xyz(samples)

    return samples


## Visualize GT or model distributions

def plot_distr(distr=None, res_npts=500, save_fig=True, model=None, device=None, base_distr=None, namestr='sphere_model'):
    on_mani, lonlat, log_detjac = make_grid_sphere(res_npts)

    if distr == '1wrapped':
        probs = true_1wrapped_probs(lonlat)
    elif distr == '4wrapped':
        probs = true_4wrapped_probs(lonlat)
    elif distr == 'bigcheckerboard':
        probs = true_bigcheckerboard_probs(lonlat)
    elif distr == 'model':
        probs = model_probs(model, on_mani, log_detjac, device, base_distr)

    plot_sphere_density(lonlat, probs, res_npts, uniform = True if distr == 'bigcheckerboard' else False)

    if save_fig:
      print(f'Saved to: {namestr}.png')
      plt.savefig(f'{namestr}.png')


def true_1wrapped_probs(lonlat):
    xyz, _ = spherical_to_xyz(lonlat)

    center = -torch.ones(3)
    center = Sphere().projx(center)
    loc = center.repeat(xyz.shape[0], 1)
    scale = torch.ones(xyz.shape[0], 2)*torch.tensor([.3,.3])
    distr = WrappedNormal(Sphere(), loc, scale)

    probs = torch.exp(distr.log_prob(xyz))
    return probs


def true_4wrapped_probs(lonlat):
    xyz, _ = spherical_to_xyz(lonlat)

    one = torch.ones(3)
    oned = torch.ones(3)
    oned[2] = -1
    centers = [one, -one, oned, -oned]
    centers = [Sphere().projx(center) for center in centers]
    n = npts*npts
    scale = torch.tensor([.3,.3])

    scales = [torch.ones(n,2)*scale for _ in range(len(centers))]
    distrs = []
    for i in range(len(centers)):
        loc = centers[i].repeat(n,1)
        distrs.append(WrappedNormal(Sphere(), loc, scales[i]))

    probs = torch.exp(distrs[0].log_prob(xyz))
    for i in range(1, len(distrs)):
        probs += torch.exp(distrs[i].log_prob(xyz))
    probs /= len(distrs)
    return probs


def true_bigcheckerboard_probs(lonlat):
    s = np.pi/2-.2 # long side length
   
    def in_board(z,s):
        # z is lonlat
        lon = z[0]
        lat = z[1]
        if np.pi <= lon < np.pi+s or np.pi-2*s <= lon < np.pi-s:
            return np.pi/2 <= lat < np.pi/2+s/2 or np.pi/2-s <= lat < np.pi/2-s/2
        elif np.pi-2*s <= lon < np.pi+2*s:
            return np.pi/2+s/2 <= lat < np.pi/2+s or np.pi/2-s/2 <= lat < np.pi/2
        else:
            return 0

    probs = torch.zeros(lonlat.shape[0])
    for i in range(lonlat.shape[0]):
        probs[i] = in_board(lonlat[i,:], s)

    probs /= torch.sum(probs)
    return probs


def model_probs(model, on_mani, log_detjac, device, base_distr):
    if device:
        on_mani = on_mani.to(device)
        log_detjac = log_detjac.to(device)

    z, logprob = model(on_mani)

    val = base_distr.log_prob(z)
    val += logprob.detach()
    val += log_detjac
    probs = torch.exp(val)
    return probs.detach()


def make_grid_sphere(npts):
    lon = torch.linspace(0, 2*np.pi, npts)
    lat = torch.linspace(0, np.pi, npts)
    Lon, Lat = torch.meshgrid((lon, lat))
    lonlat = torch.stack([Lon.flatten(), Lat.flatten()], dim=1)
    xyz, detjac = spherical_to_xyz(lonlat)
    log_detjac = torch.log(torch.abs(detjac))
    on_mani = xyz
    return on_mani, lonlat, log_detjac

