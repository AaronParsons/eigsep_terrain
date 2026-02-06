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

def calc_maxiter(E, N, U, start_point, delta_r_m=1, r_max=None, dtype=dtype_r):
    '''Return the maximum iterations needed to resolve distances for ray tracing.'''
    if r_max is None:
        corners = np.array([[E[0],  N[0],  U[0, 0]],
                            [E[0],  N[-1], U[-1, 0]],
                            [E[-1], N[0],  U[0, -1]],
                            [E[-1], N[-1], U[-1, -1]]], dtype=dtype)
        r_max = np.linalg.norm(corners - start_point[None, :], axis=1).max()
    max_iter = int(np.ceil(r_max / delta_r_m))
    return max_iter

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
        max_iter = calc_maxiter(E, N, U, start_point, delta_r_m=delta_r_m, r_max=r_max, dtype=dtype)
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

def ray_trace_basic_jax(E, N, U, start_point, rays, delta_r_m=1.0,
                        r_start=None, max_iter=4096):
    E = jnp.asarray(E)
    N = jnp.asarray(N)
    U = jnp.asarray(U)
    start_point = jnp.asarray(start_point)
    rays = jnp.asarray(rays)

    Ne = E.shape[0]
    Nn = N.shape[0]
    Nr = rays.shape[1]

    E0, Emax = E[0], E[-1]
    N0, Nmax = N[0], N[-1]

    dE = E[1] - E[0]
    dN = N[1] - N[0]
    inv_dE = 1.0 / dE
    inv_dN = 1.0 / dN

    u_max = jnp.max(U)

    if r_start is None:
        r0 = jnp.full((Nr,), delta_r_m, dtype=U.dtype)
        active0 = jnp.ones((Nr,), dtype=jnp.bool_)
    else:
        r0 = jnp.asarray(r_start, dtype=U.dtype)
        active0 = ~jnp.isnan(r0)

    def step_fn(state):
        i, r, active = state

        pts = start_point[:, None] + r[None, :] * rays
        px, py, pz = pts[0], pts[1], pts[2]

        in_bounds = (px >= E0) & (px <= Emax) & (py >= N0) & (py <= Nmax)

        e_px = jnp.floor((px - E0) * inv_dE).astype(jnp.int32)
        n_px = jnp.floor((py - N0) * inv_dN).astype(jnp.int32)
        e_px_c = jnp.clip(e_px, 0, Ne - 1)
        n_px_c = jnp.clip(n_px, 0, Nn - 1)

        u_m = U[n_px_c, e_px_c]

        live = active & in_bounds

        # hit: stop and keep r (matches NumPy prune-by-(pz>u_m))
        hit = live & (pz <= u_m)

        # above-ground candidates (before sentinel filtering)
        above = live & (pz > u_m)

        # NumPy semantics: if above-ground but u_m is "invalid" (== u_max), set NaN
        invalid = above & ~(u_m < u_max)

        # continue only if above-ground and valid terrain
        cont = above & (u_m < u_max)

        # out-of-bounds -> NaN
        oob = active & (~in_bounds)

        # step continuing rays
        r = jnp.where(cont, r + delta_r_m, r)

        # apply NaNs (newly OOB or invalid terrain while above ground)
        r = jnp.where(oob | invalid, jnp.nan, r)

        # keep only continuing rays active
        active = cont

        return (i + 1, r, active)

    def cond_fn(state):
        i, r, active = state
        return (i < max_iter) & jnp.any(active)

    _, r_final, active_final = lax.while_loop(cond_fn, step_fn, (jnp.int32(0), r0, active0))

    # remaining active => never hit by max_iter => NaN
    r_final = jnp.where(active_final, jnp.nan, r_final)
    return r_final

# JIT-compiled callable (max_iter treated as static if you wrap it this way)
ray_trace_basic_jax_jit = jax.jit(ray_trace_basic_jax, static_argnames=("max_iter",))

