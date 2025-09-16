"""Tests for eigsep_corr.io"""
import copy
import os
import pytest
import numpy as np
import healpy
from eigsep_terrain.ray import ray_trace_dda_jax, ray_trace_basic


class TestUtils:
    def test_ray_trace_dda_jax(self):
        '''Unit test for calc_az_bin_range'''
        npnts = 512
        E = np.linspace(-1000, 1000, npnts)
        N = np.linspace(-1000, 1000, npnts)
        U = np.ones((npnts, npnts), dtype=float)
        start_point = np.array([0, 0, 2], dtype=float)
        nside = 128
        r = ray_trace_dda_jax(E, N, U, start_point, nside)
        npix = healpy.nside2npix(nside)
        px = np.arange(npix)
        th, phi = healpy.pix2ang(nside, ipix=px)
        assert np.all(np.isnan(r[th < np.pi/2]))
        assert np.all(r[th > np.pi/2] >= 1)

    def test_ray_trace_basic(self):
        '''Unit test for calc_az_bin_range'''
        npnts = 512
        E = np.linspace(-1000, 1000, npnts)
        N = np.linspace(-1000, 1000, npnts)
        U = np.ones((npnts, npnts), dtype=float)
        start_point = np.array([0, 0, 2], dtype=float)
        nside = 128
        r = ray_trace_basic(E, N, U, start_point, nside)
        npix = healpy.nside2npix(nside)
        px = np.arange(npix)
        th, phi = healpy.pix2ang(nside, ipix=px)
        assert np.all(np.isnan(r[th < np.pi/2]))
        assert np.all(r[th > np.pi/2] >= 1)
