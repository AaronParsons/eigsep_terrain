import numpy as np
import healpy
import jax
import jax.numpy as jnp
from jax import lax
from .utils import distance

dtype_r = np.float32
dtype_i = np.int32
eps = dtype_r(1e-8)
zero = dtype_r(0.0)
big = dtype_r(1e30)

def ray_triangle_intersect_batch(orig, dirv, v0, v1, v2):
    """
    Vectorized Möller–Trumbore.
    orig, dirv: (..., 3)
    v0,v1,v2 : (..., 3)
    Returns (hit: (...,), t: (...,)) with t >= 0 when hit; no backface cull.
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = jnp.cross(dirv, edge2)
    det  = jnp.sum(edge1 * pvec, axis=-1)
    inv_det = 1.0 / det
    is_pl = jnp.abs(det) < eps  # detect if parallel
    tvec = orig - v0
    u = jnp.sum(tvec * pvec, axis=-1) * inv_det
    qvec = jnp.cross(tvec, edge1)
    v = jnp.sum(dirv * qvec, axis=-1) * inv_det
    t = jnp.sum(edge2 * qvec, axis=-1) * inv_det
    hit = (~is_pl) & (u >= 0.0) & (u <= 1.0) & \
          (v >= 0.0) & (u + v <= 1.0) & (t > 0.0)
    return hit, t

# ---------- Main JAX DDA tracer over all HEALPix rays ----------
def ray_trace_dda_jax(E, N, U, start_point, nside,
                      r_max=None, max_horizon_ang_deg=45.0):
    """
    JAX-accelerated DDA heightfield intersector.
    Returns distances in HEALPix order; NaN for no hit within r_max or above horizon.
    Assumes uniform E/N spacing.
    """
    start_point = np.asarray(start_point, dtype=dtype_r)
    start = jnp.asarray(start_point)  # (3,)
    shape = ny, nx = U.shape
    if r_max is None:
        corners = np.array([[E[0],  N[0],  U[0, 0]],
                            [E[0],  N[-1], U[-1, 0]],
                            [E[-1], N[0],  U[0, -1]],
                            [E[-1], N[-1], U[-1, -1]]], dtype=dtype_r)
        r_max = np.linalg.norm(corners - start_point[None,:], axis=1).max()
    r_max = dtype_r(r_max)

    U = jnp.asarray(U, dtype=dtype_r)  # (ny, nx) on device
    E0 = dtype_r(E[0])
    N0 = dtype_r(N[0])
    dE = dtype_r(E[1] - E0)
    dN = dtype_r(N[1] - N0)

    # Directions from HEALPix
    npix = healpy.nside2npix(nside)
    px = np.arange(npix)
    _dirs = np.asarray(healpy.pix2vec(nside, px)).transpose()
    dirs = jnp.asarray(_dirs)

    # Horizon cull
    max_horizon_cos = np.cos(np.deg2rad(max_horizon_ang_deg)).astype(dtype_r)
    dz = dirs[:, 2]
    # keep only rays with theta > horizon => dz < cos(theta_max). Others set to NaN at end.
    active0 = dz < max_horizon_cos

    # Initial DDA per-ray parameters
    dx, dy, dz = dirs[:,0], dirs[:,1], dirs[:,2]

    # Integer cell indices from start
    # Same start for all rays
    i0 = jnp.floor((start[0] - E0) / dE).astype(dtype_i)
    j0 = jnp.floor((start[1] - N0) / dN).astype(dtype_i)
    i = jnp.full((npix,), i0, dtype=dtype_i)
    j = jnp.full((npix,), j0, dtype=dtype_i)


    # stepX, tMaxX, tΔX
    stepX = jnp.where(dx > 0, 1, jnp.where(dx < 0, -1, 0)).astype(dtype_i)
    nextEx = jnp.where(dx > 0, E0 + (i.astype(dtype_r) + 1.0)*dE,
                       jnp.where(dx < 0, E0 + (i.astype(dtype_r))*dE, zero))
    tMaxX  = jnp.where(dx > 0, (nextEx - start[0]) / dx,
             jnp.where(dx < 0, (nextEx - start[0]) / dx, big))
    tΔX = jnp.where(dx != 0, jnp.abs(dE / dx), big)

    # stepY, tMaxY, tΔY
    stepY = jnp.where(dy > 0, 1, jnp.where(dy < 0, -1, 0)).astype(dtype_i)
    nextNy = jnp.where(dy > 0, N0 + (j.astype(dtype_r) + 1.0)*dN,
                       jnp.where(dy < 0, N0 + (j.astype(dtype_r))*dN, zero))
    tMaxY  = jnp.where(dy > 0, (nextNy - start[1]) / dy,
             jnp.where(dy < 0, (nextNy - start[1]) / dy, big))
    tΔY = jnp.where(dy != 0, jnp.abs(dN / dy), big)

    # Carry state
    tPrev = jnp.zeros((npix,), dtype=dtype_r)
    out_r = jnp.full((npix,), jnp.nan, dtype=dtype_r)
    active = active0

    def cond(state):
        return jnp.any(state[-1])

    def body(state):
        i, j, tMaxX, tMaxY, tΔX, tΔY, stepX, stepY, tPrev, out_r, active = state

        # Next boundary (param) for each ray
        tNext = jnp.minimum(tMaxX, tMaxY)
        tNext = jnp.minimum(tNext, r_max)

        # Valid cells (for triangle test): 0<=i<nx-1, 0<=j<ny-1
        valid_cell = (i >= 0) & (j >= 0) & (i < (nx-1)) & (j < (ny-1))

        test_mask = active & valid_cell & (tPrev <= r_max)

        # Build triangles only where test_mask; elsewhere, fill with dummies
        Ex0 = E0 + i.astype(dtype_r) * dE
        Ex1 = Ex0 + dE
        Ny0 = N0 + j.astype(dtype_r) * dN
        Ny1 = Ny0 + dN

        # sample z's
        # safe gather by clamping to grid for masked elements (mask blocks their use)
        ii = jnp.clip(i, 0, nx-2)
        jj = jnp.clip(j, 0, ny-2)
        u00 = U[jj, ii]
        u10 = U[jj, ii+1]
        u01 = U[jj+1, ii]
        u11 = U[jj+1, ii+1]

        v00 = jnp.stack([Ex0, Ny0, u00], axis=-1)
        v10 = jnp.stack([Ex1, Ny0, u10], axis=-1)
        v01 = jnp.stack([Ex0, Ny1, u01], axis=-1)
        v11 = jnp.stack([Ex1, Ny1, u11], axis=-1)

        # Broadcast start/dirs to (npix,3)
        starts = jnp.broadcast_to(start, v00.shape)

        # Two triangles: (v00, v10, v01) and (v11, v10, v01)
        hit1, t1 = ray_triangle_intersect_batch(starts, dirs, v00, v10, v01)
        hit2, t2 = ray_triangle_intersect_batch(starts, dirs, v11, v10, v01)

        # Pick min positive t
        t_hit = jnp.where(hit1, t1, jnp.inf)
        t_hit = jnp.minimum(t_hit, jnp.where(hit2, t2, jnp.inf))

        # Accept only within current segment [tPrev, tNext]
        hit_now = test_mask & jnp.isfinite(t_hit) & (t_hit >= tPrev) & (t_hit <= tNext)

        # Write results where first time we hit (keep previous out_r where already set)
        out_r = jnp.where(hit_now & jnp.isnan(out_r), t_hit.astype(dtype_r), out_r)

        # Rays that remain active: not hit, not exceeded r_max this step
        still_active = active & (~hit_now) & (tNext < r_max)

        # Advance DDA to next boundary for still-active rays
        chooseX = tMaxX < tMaxY
        tPrev = jnp.where(still_active, tNext, tPrev)

        # Step i/j and tMaxX/Y selectively
        i = jnp.where(still_active & chooseX, i + stepX, i)
        j = jnp.where(still_active & (~chooseX), j + stepY, j)
        tMaxX = jnp.where(still_active & chooseX, tMaxX + tΔX, tMaxX)
        tMaxY = jnp.where(still_active & (~chooseX), tMaxY + tΔY, tMaxY)

        # Deactivate rays that reached r_max this step without hit
        active = still_active

        return (i, j, tMaxX, tMaxY, tΔX, tΔY, stepX, stepY, tPrev, out_r, active)

    state0 = (i, j, tMaxX, tMaxY, tΔX, tΔY, stepX, stepY, tPrev, out_r, active)
    out_r = lax.while_loop(cond, body, state0)[-2]

    # Rays above horizon or upward-pointing -> NaN
    out_r = jnp.where(active0, out_r, jnp.nan)

    #return np.asarray(out_r)
    return out_r


def ray_trace_basic(E, N, U, start_point, nside, delta_r_m=1,
                        r_max=None, max_horizon_ang_deg=45, dtype=dtype_r):
    '''Return the distance along a HealPix grid of specified nside from a
    ENU starting point until a ray intersects the terrain, in steps of 
    delta_r_m [m], out to a specified r_max_m (or map edge, if None).
    Don't bother checking above the specified max_horizon_ang_deg [deg],
    as these points are assumed not to intersect terrain. Returns distance
    [m] in HealPix order, with non-intersecting points set to NaN.'''
    if r_max is None:
        r_max = 0
        for e_ind in (0, -1):
            for n_ind in (0, -1):
                r_max = max([r_max, distance(start_point, np.array([E[e_ind], N[n_ind], U[n_ind, e_ind]]))])
    E_res = E[1] - E[0]
    N_res = N[1] - N[0]
    max_iter = np.ceil(r_max / delta_r_m).astype(int)
    npix = healpy.nside2npix(nside)
    px = np.arange(npix)
    dr_vec = (delta_r_m * np.array(healpy.pix2vec(nside, px))).astype(dtype)
    th, _ = healpy.pix2ang(nside, px)
    r = delta_r_m * np.ones(npix, dtype=dtype)
    tracing = th > np.deg2rad(max_horizon_ang_deg)
    inds = np.where(tracing)[0]
    r[~tracing] = np.nan
    points_m = start_point[:, None] + dr_vec[:, inds]
    for i in range(2, max_iter):
        e_px = np.floor((points_m[0] - E[0]) / E_res).astype(int)
        n_px = np.floor((points_m[1] - N[0]) / N_res).astype(int)
        u_m = U[n_px, e_px]
        # prune to points that are above ground
        tracing = (points_m[2] > u_m)
        inds = inds[tracing]
        if inds.size == 0:
            break
        # take a step along the ray
        r[inds] = i * delta_r_m
        points_m = points_m[:, tracing] + dr_vec[:, inds]
        # check if out of bounds
        tracing = np.logical_and(E[0] <= points_m[0], points_m[0] <= E[-1])
        tracing &= np.logical_and(N[0] <= points_m[1], points_m[1] <= N[-1])
        r[inds[~tracing]] = np.nan  # set newly out-of-bounds rays to nan
        # prune to points that are in bounds
        inds = inds[tracing]
        points_m = points_m[:, tracing]
        if inds.size == 0:
            break
    # Any remaining active pixels should be set to nan
    r[inds] = np.nan
    return r
