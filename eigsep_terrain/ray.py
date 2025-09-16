import numpy as np
import healpy
import jax
import jax.numpy as jnp
from jax import lax
from .utils import distance

dtype_r = np.float32
dtype_i = np.int32

def ray_trace_basic(E, N, U, start_point, ray=None, nside=None, delta_r_m=1,
                        r_max=None, dtype=dtype_r):
    '''Return the distance along a HealPix grid of specified nside from a
    ENU starting point until a ray intersects the terrain, in steps of 
    delta_r_m [m], out to a specified r_max_m (or map edge, if None).
    Don't bother checking above the specified max_horizon_ang_deg [deg],
    as these points are assumed not to intersect terrain. Returns distance
    [m] in HealPix order, with non-intersecting points set to NaN.'''

    if r_max is None:
        corners = np.array([[E[0],  N[0],  U[0, 0]],
                            [E[0],  N[-1], U[-1, 0]],
                            [E[-1], N[0],  U[0, -1]],
                            [E[-1], N[-1], U[-1, -1]]], dtype=dtype)
        r_max = np.linalg.norm(corners - start_point[None, :], axis=1).max()
    r_max = dtype(r_max)
    u_max = dtype(U.max())
    res = E[1] - E[0]
    assert np.abs(res - (N[1] - N[0])) < 1e-5
    max_iter = int(np.ceil(r_max / delta_r_m))
    if ray is None:
        assert nside is not None
        npix = healpy.nside2npix(nside)
        px = np.arange(npix)
        ray = np.array(healpy.pix2vec(nside, px))
    else:
        assert ray.shape[0] == 3
        ray /= np.linalg.norm(ray, axis=0)
    dr_vec = (delta_r_m * ray).astype(dtype)
    r = delta_r_m * np.ones(npix, dtype=dtype)
    active = np.ones(npix, dtype=bool)
    inds = np.flatnonzero(active)
    r[~active] = np.nan
    points_m = start_point[:, None] + dr_vec[:, inds]
    for i in range(2, max_iter):
        e_px = np.floor((points_m[0] - E[0]) / res).astype(dtype_i)
        n_px = np.floor((points_m[1] - N[0]) / res).astype(dtype_i)
        u_m = U[n_px, e_px]
        # prune to points that are above ground
        active = (points_m[2] > u_m)
        inds = inds[active]
        if inds.size == 0:
            break
        # take a step along the ray
        r[inds] = i * delta_r_m
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
