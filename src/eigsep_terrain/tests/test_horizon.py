import numpy as np
import pytest

from eigsep_terrain.dem import DEM


def _synthetic_dem(data, res=1.0):
    dem = DEM()
    dem.res = res
    dem.data = np.asarray(data)
    dem.map_crd = {'eastbc': 0, 'westbc': 0, 'northbc': 0, 'southbc': 0}
    dem.survey_offset = np.array([0, 0, 0])
    return dem


def _brute_force_horizon(dem, e0, n0, u0, n_az):
    '''Bin maxima over every full-resolution pixel, with no pyramid.

    A pixel touches every bin from the bin of its lowest corner azimuth
    to the bin of its highest, both included; the observer's own pixel
    touches all bins.'''
    res = dem.res
    d_az = 2 * np.pi / n_az
    hangles = np.zeros(n_az)
    crds = np.zeros((2, n_az))
    for ni, ei in np.ndindex(dem.data.shape):
        e_lo, e_hi = (ei - 0.5) * res - e0, (ei + 0.5) * res - e0
        n_lo, n_hi = (ni - 0.5) * res - n0, (ni + 0.5) * res - n0
        de = max(e_lo, 0, -e_hi)
        dn = max(n_lo, 0, -n_hi)
        h = np.arctan2(dem.data[ni, ei] - u0, np.hypot(de, dn))
        if de == 0 and dn == 0:
            bins = np.arange(n_az)
        else:
            az = np.arctan2([e_lo, e_lo, e_hi, e_hi],
                            [n_lo, n_hi, n_lo, n_hi]) % (2 * np.pi)
            b = np.around(az / d_az).astype(int) % n_az
            rel = (b - b[0] + n_az // 2) % n_az - n_az // 2
            bins = (b[0] + np.arange(rel.min(), rel.max() + 1)) % n_az
        update = hangles[bins] < h
        hangles[bins] = np.where(update, h, hangles[bins])
        crds[0, bins] = np.where(update, ni * res, crds[0, bins])
        crds[1, bins] = np.where(update, ei * res, crds[1, bins])
    return hangles, crds


def test_calc_horizon_fills_last_bin_of_straddling_pixel():
    # One 1 m pixel 10 m east of the observer spans azimuths
    # 90 -/+ atan(0.5 / 9.5) = 86.99..93.01 deg, so 1 deg bins 87..93.
    dem = _synthetic_dem([[10.0]])

    hangles, _ = dem.calc_horizon(-10.0, 0.0, 0.0, n_az=360,
                                  azimuth_frame='grid')

    np.testing.assert_array_equal(np.flatnonzero(hangles > 0),
                                  np.arange(87, 94))


def test_calc_horizon_fills_last_bin_across_wrap():
    # The same pixel 10 m north spans -3.01..+3.01 deg: bins 357..3.
    dem = _synthetic_dem([[10.0]])

    hangles, _ = dem.calc_horizon(0.0, -10.0, 0.0, n_az=360,
                                  azimuth_frame='grid')

    np.testing.assert_array_equal(np.flatnonzero(hangles > 0),
                                  [0, 1, 2, 3, 357, 358, 359])


@pytest.mark.parametrize('n_az', [360, 1440])
@pytest.mark.parametrize('observer', [(6.3, 8.1, 2.0), (-3.7, 20.2, 1.0)])
def test_calc_horizon_pyramid_matches_brute_force(observer, n_az):
    # 37 x 29 is not a multiple of the pooling factor, so the pyramid
    # has padded partial blocks; the observers sit inside and outside.
    rng = np.random.default_rng(1)
    dem = _synthetic_dem(rng.uniform(0, 5, size=(37, 29)), res=0.5)

    hangles, crds = dem.calc_horizon(*observer, n_az=n_az,
                                     azimuth_frame='grid')
    want_h, want_crds = _brute_force_horizon(dem, *observer, n_az)

    np.testing.assert_allclose(hangles, want_h, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(crds, want_crds)


def test_calc_horizon_crds_preserve_submeter_coordinates():
    dem = DEM()
    dem.res = 0.5
    dem.data = np.array([[-100, -100], [-100, 100]], dtype=np.float32)
    dem.map_crd = {'eastbc': 0, 'westbc': 0, 'northbc': 0, 'southbc': 0}
    dem.survey_offset = np.array([0, 0, 0])

    hangles, crds = dem.calc_horizon(0.0, 0.0, 0.0, n_az=64)

    assert np.issubdtype(crds.dtype, np.floating)
    assert np.any(np.isclose(crds[0], 0.5) & np.isclose(crds[1], 0.5))
    assert np.any(hangles > 0)
