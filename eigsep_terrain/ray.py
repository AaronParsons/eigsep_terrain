import numpy as np
import healpy
import jax
import jax.numpy as jnp
from jax import lax
from .utils import distance

dtype_r = np.float32
dtype_i = np.int32

def healpix_rays(nside, dtype=dtype_r):
    npix = healpy.nside2npix(nside)
    px = np.arange(npix)
    rays = np.array(healpy.pix2vec(nside, px), dtype=dtype)
    return rays

def ray_trace_basic(E, N, U, start_point, rays, delta_r_m=1,
                    r_start=None, max_iter=None, r_max=None, dtype=dtype_r):
    '''Return the distance along a HealPix grid of specified nside from a
    ENU starting point until a ray intersects the terrain, in steps of 
    delta_r_m [m], out to a specified r_max_m (or map edge, if None).
    Don't bother checking above the specified max_horizon_ang_deg [deg],
    as these points are assumed not to intersect terrain. Returns distance
    [m] in HealPix order, with non-intersecting points set to NaN.'''

    E = np.asarray(E, dtype=dtype)
    N = np.asarray(N, dtype=dtype)
    U = np.asarray(U, dtype=dtype, order='C')
    start_point = np.asarray(start_point, dtype=dtype)
    rays = np.asarray(rays, dtype=dtype)
    delta_r_m = dtype(delta_r_m)
    u_max = dtype(U.max())
    dE = E[1] - E[0]
    dN = N[1] - N[0]
    if max_iter is None:
        if r_max is None:
            corners = np.array([[E[0],  N[0],  U[0, 0]],
                                [E[0],  N[-1], U[-1, 0]],
                                [E[-1], N[0],  U[0, -1]],
                                [E[-1], N[-1], U[-1, -1]]], dtype=dtype)
            r_max = np.linalg.norm(corners - start_point[None, :], axis=1).max()
        max_iter = int(np.ceil(r_max / delta_r_m))
    assert rays.shape[0] == 3
    dr_vec = delta_r_m * rays
    if r_start is None:
        r = np.full(rays.shape[1:], delta_r_m, dtype=dtype)
    else:
        r = np.asarray(r_start, dtype=dtype)
    active = ~np.isnan(r)
    inds = np.flatnonzero(active)
    points_m = start_point[:, None] + r[None, inds] * rays[:, inds]
    for i in range(2, max_iter):
        e_px = np.floor((points_m[0] - E[0]) / dE).astype(dtype_i)
        n_px = np.floor((points_m[1] - N[0]) / dN).astype(dtype_i)
        u_m = U[n_px, e_px]
        # prune to points that are above ground
        active = (points_m[2] > u_m)
        inds = inds[active]
        #print(i, np.sum(active), inds[:10])
        if inds.size == 0:
            break
        # take a step along the ray
        r[inds] += delta_r_m
        points_m = points_m[:, active] + dr_vec[:, inds]
        # check if out of bounds
        active  = (u_m[active] < u_max)
        active &= (E[0] <= points_m[0]) & (points_m[0] <= E[-1])
        active &= (N[0] <= points_m[1]) & (points_m[1] <= N[-1])
        r[inds[~active]] = np.nan  # set newly out-of-bounds rays to nan
        # prune to points that are in bounds
        inds = inds[active]
        points_m = points_m[:, active]
        if inds.size == 0:
            break
    # Any remaining active pixels should be set to nan
    r[inds] = np.nan
    return r
