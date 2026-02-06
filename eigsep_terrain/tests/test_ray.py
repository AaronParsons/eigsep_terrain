"""Tests for eigsep_corr.io"""
import copy
import os
import pytest
import numpy as np
import healpy
from eigsep_terrain.ray import ray_trace_basic_jax, ray_trace_basic, healpix_rays, calc_maxiter, ray_trace_basic_jax_jit


class TestUtils:

    def test_ray_trace_basic(self):
        '''Unit test for calc_az_bin_range'''
        npnts = 512
        E = np.linspace(-1000, 1000, npnts)
        N = np.linspace(-1000, 1000, npnts)
        U = np.ones((npnts, npnts), dtype=float)
        start_point = np.array([0, 0, 2], dtype=float)
        nside = 128
        rays = healpix_rays(nside)
        rays_2d = rays.reshape(rays.shape[0], -1) 
        r = ray_trace_basic(E, N, U, start_point, rays_2d, nside)
        npix = healpy.nside2npix(nside)
        px = np.arange(npix)
        th, phi = healpy.pix2ang(nside, ipix=px)
        assert np.all(np.isnan(r[th < np.pi/2]))
        assert np.all(r[th > 1.01 * np.pi/2] >= 1)

    def test_ray_trace_basic_jax(self):
        '''Unit test for calc_az_bin_range'''
        npnts = 512
        E = np.linspace(-1000, 1000, npnts)
        N = np.linspace(-1000, 1000, npnts)
        U = np.ones((npnts, npnts), dtype=float)
        start_point = np.array([0, 0, 2], dtype=float)
        nside = 128
        rays = healpix_rays(nside)
        rays_2d = rays.reshape(rays.shape[0], -1) 
        #max_iter = calc_maxiter(E, N, U, start_point)
        max_iter = 4096
        r = ray_trace_basic_jax_jit(E, N, U, start_point, rays_2d, nside, max_iter=max_iter)
        npix = healpy.nside2npix(nside)
        px = np.arange(npix)
        th, phi = healpy.pix2ang(nside, ipix=px)
        assert np.all(np.isnan(r[th < np.pi/2]))
        assert np.all(r[th > 1.01 * np.pi/2] >= 1)
