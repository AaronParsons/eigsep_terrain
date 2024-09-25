'''Tools for dealing with digital elevation models.'''

import numpy as np
import PIL.Image
import os
import pyuvdata
import xmltodict
from .utils import *

XML_CRD_KEYWORDS = ('eastbc', 'westbc', 'northbc', 'southbc')

class DEM(dict):
    '''Class for interacting with Digital Elevation Model data.'''
    
    def __init__(self, cache_file=None, clear_cache=False):
        self._cache_file = cache_file
        if clear_cache and os.path.exists(cache_file):
            os.remove(cache_file)
        if cache_file is not None and os.path.exists(cache_file):
            self.load_cache()

    def load_cache(self):
        '''Retrieve cached DEM data from npz file.'''
        npz = np.load(self._cache_file)
        self.files = npz['files']
        self.res = npz['res']
        self.data = npz['dem']
        self.map_crd = {k: npz[k] for k in XML_CRD_KEYWORDS}
        self.survey_offset = npz['survey_offset']

    def save_cache(self):
        '''Cache DEM data in npz file.'''
        if self._cache_file is not None:
            np.savez(self._cache_file, dem=self.data, res=self.res,
                     files=self.files, survey_offset=self.survey_offset,
                     **self.map_crd)

    def load_tif(self, files, survey_offset=np.array([0, 0,0])):
        _dem = np.hstack([np.vstack([np.array(PIL.Image.open(f),
                                              dtype='int32')
                                     for f in files[i][::-1]])
                          for i in range(files.shape[0])])
        self.files = files
        self.res = 1000 / 2000 # m / px
        self.data = np.flipud(_dem)
        self.survey_offset = survey_offset

    def load_xml(self, filename):
        with open(filename, 'rb') as f:
           self.map_crd = {k: np.deg2rad(float(v)) for k, v in
     xmltodict.parse(f)['metadata']['idinfo']['spdom']['bounding'].items()}

    def latlon_to_enu(self, lat_str, lon_str, alt_str=None):
        '''Convert lat/lon/[alt] deg/deg/[m] strings into east/north/up
        coordinates in meters.'''
        lat = np.deg2rad(float(lat_str))
        lon = np.deg2rad(float(lon_str))
        if alt_str is None:
            alt = 0
        else:
            alt = float(alt_str)
        ecef = pyuvdata.utils.XYZ_from_LatLonAlt(lat, lon, 0)
        # XXX don't understand negative alt below
        enu = pyuvdata.utils.ENU_from_ECEF(ecef,
                latitude=self.map_crd['southbc'],
                longitude=self.map_crd['westbc'], altitude=-alt)
        return enu - self.survey_offset

    def m2px(self, *args):
        px = tuple(np.around(m / self.res).astype(int) for m in args)
        return px
        
    def interp_alt(self, e_m, n_m, return_vec=False):
        e_px, n_px = self.m2px(e_m, n_m)
        u_m = self.data[n_px, e_px]
        if return_vec:
            try:
                return np.concatenate([e_m, n_m, u_m], axis=0)
            except(ValueError):
                return np.array([e_m, n_m, u_m])
        else:
            return u_m

    def add_survey_points(self, pnts):
        self.update({k: self.latlon_to_enu(*v.split(', '))
                     for k, v in pnts.items()})

    def get_en(self, erng_m=None, nrng_m=None, return_px=False,
                     decimate=1, edges=False):
        if erng_m == None:
            emn, emx = 0, self.data.shape[1]
        else:
            emn, emx = self.m2px(*erng_m)
        if nrng_m == None:
            nmn, nmx = 0, self.data.shape[0]
        else:
            nmn, nmx = self.m2px(*nrng_m)
        if edges:
            _E = np.arange(emn, emx + decimate, decimate) - 0.5
            _N = np.arange(nmn, nmx + decimate, decimate) - 0.5
        else:
            _E = np.arange(emn, emx, decimate)
            _N = np.arange(nmn, nmx, decimate)
        if return_px:
            return _E, _N
        else:
            return _E * self.res, _N * self.res
            
    def get_tile(self, erng_m=None, nrng_m=None, mesh=True):
        _E, _N = self.get_en(erng_m, nrng_m, return_px=True)
        U = self.data[_N[0]:_N[-1]+1, _E[0]:_E[-1]+1]
        if mesh:
            E, N = np.meshgrid(_E, _N)
        else:
            E, N = _E, _N
        return E * self.res, N * self.res, U

    def find_anchors(self, e0, n0, u0, n_anchors=2, r_anchor_max=300,
                     min_angle=np.deg2rad(20), n_az_bins=240):
        '''Find opposing anchor positions.
        r_anchor_max: meters, min_angle: radians.'''
        # Find anchor points
        erng = (e0 - r_anchor_max, e0 + r_anchor_max)
        nrng = (n0 - r_anchor_max, n0 + r_anchor_max)
        E, N, U = self.get_tile(erng, nrng, mesh=False)
        rdist = np.sqrt((E[None, :] - e0)**2 + (N[:, None] - n0)**2)
        cone = u0 + np.tan(min_angle) * rdist
        inds = (U - cone > 0)
        rmin = r_anchor_max * np.ones(n_az_bins)
        rmax = np.zeros(n_az_bins)
        b = az_bin(E - e0, N - n0, n_az_bins)
        for _r, _b in zip(rdist[inds], b[inds]):
            rmin[_b % n_az_bins] = min(rmin[_b % n_az_bins], _r)
            rmax[_b % n_az_bins] = max(rmax[_b % n_az_bins], _r)
        # Assign inf to areas that don't meet anchor length requirements
        rmin = np.where(rmin >= r_anchor_max, np.Inf, rmin)
        # Fold to enforce anchors being on opposite sides,
        # then minimize total anchor length
        rmin.shape = (n_anchors, -1)
        rmax.shape = (n_anchors, -1)
        rtot = np.sum(rmin, axis=0)
        bmin = np.argmin(rtot)
        r_anchors = rmin[:, bmin]
        az_min = bmin * 2 * np.pi / n_az_bins
        az_anchors = az_min + 2 * np.pi / n_anchors * np.arange(n_anchors)
        anchors_e = e0 + r_anchors * np.sin(az_anchors)
        anchors_n = n0 + r_anchors * np.cos(az_anchors)
        # find boundary
        valid = np.where(rtot < n_anchors * r_anchor_max)[0]
        az_valid = valid * 2 * np.pi / n_az_bins
        boundary = []
        for a in range(n_anchors):
            bound_min = [(e0 + r * np.sin(az), n0 + r * np.cos(az))
                         for r, az in zip(rmin[a, valid],
                                     az_valid + a * 2 * np.pi / n_anchors)]
            bound_max = [(e0 + r * np.sin(az), n0 + r * np.cos(az))
                         for r, az in zip(rmax[a, valid],
                                     az_valid + a * 2 * np.pi / n_anchors)]
            boundary.append(bound_min + bound_max[::-1])

        return list(zip(anchors_e, anchors_n)), np.array(boundary)

    def build_maxpool_pyramid(self, data=None, factor=4):
        '''Return a list of (2D array, factor) pairs, each downsampled
        along 2 dimensions by the specified factor and maxpooled.'''
        if data is None:
            data = self.data
        answer = [(data, 1)]
        # trim off remainders
        r0 = data.shape[0] % factor
        r1 = data.shape[1] % factor
        data = data[:data.shape[0]-r0, :data.shape[1]-r1]
        if data.shape[0] <= factor or data.shape[1] <= factor:
            return answer
        data.shape = (data.shape[0] // factor, factor,
                      data.shape[1] // factor, factor)
        pool_data = np.max(data, axis=(1, 3))
        if r1 > 0:
            # pad out fractional pixel on boundary
            pool_data = np.concatenate([pool_data,
                            np.zeros_like(pool_data[:,:1])], axis=1)
        if r0 > 0:
            # pad out fractional pixel on boundary
            pool_data = np.concatenate([pool_data,
                            np.zeros_like(pool_data[:1])], axis=0)
        answer += [(d, factor * f) for (d, f) in
                    self.build_maxpool_pyramid(data=pool_data,
                                               factor=factor)]
        return answer

    def calc_horizon(self, e0, n0, u0, n_az=256, imp=None, f_prev=None,
                     ei_off=None, ni_off=None, crds=None,
                     hangles=None):
        if imp is None:
            # top case
            imp = self.build_maxpool_pyramid()
            if hangles is None:
                hangles = np.zeros(n_az)
            if crds is None:
                crds = np.zeros([2, hangles.size], dtype=int)
            else:
                crds_px = np.around(crds / self.res).astype(int)
                _U, _f = imp[0]
                e_edges, n_edges = self.get_en(edges=True, decimate=_f)
                n_edges = n_edges[crds_px[0]]
                e_edges = e_edges[crds_px[1]]
                r_min = np.sqrt((e_edges - e0)**2 + (n_edges - n0)**2)
                az = np.arctan2(e_edges - e0, n_edges - n0)
                az = np.where(az < 0, 2 * np.pi + az, az)
                b = np.around(az / (2 * np.pi / n_az)).astype(int)
                hangles[b % n_az] = np.arctan2(_U[crds_px[0], crds_px[1]]
                                               - u0, r_min)
            U, f = imp[-1]
            e_edges, n_edges = self.get_en(edges=True, decimate=f)
            _ni, _ei = 0, 0
        else:
            _U, f = imp[-1]
            f_step = (f_prev // f)
            _ni, _ei = ni_off * f_step, ei_off * f_step
            _e_edges, _n_edges = self.get_en(edges=True, decimate=f)
            n_edges = _n_edges[_ni:_ni + f_step + 1]
            e_edges = _e_edges[_ei:_ei + f_step + 1]
            U = _U[_ni:_ni + f_step, _ei:_ei + f_step]
    
        r_min = calc_rmin(e_edges, n_edges, e0, n0)
        az_min, az_max = calc_az_bin_range(e_edges, n_edges, e0, n0, n_az)
        hor_ang = np.arctan2(U - u0, r_min)
        # process in order of maximum possible horizon angle first
        n_pxs, e_pxs = np.unravel_index(np.argsort(-hor_ang, axis=None),
                                        r_min.shape)
        for cnt, (ni, ei) in enumerate(zip(n_pxs, e_pxs)):
            bmin = az_min[ni, ei]
            bmax = az_max[ni, ei]
            h = hor_ang[ni, ei]
            if bmin < bmax:
                slices = [slice(bmin, bmax)]
            elif bmin == bmax:
                slices = [slice(bmin, bmin+1)]
            else:
                slices = [slice(bmin, None), slice(0, bmax)]
            if len(imp) == 1:
                # base case
                for s in slices:
                    update = (hangles[s] < h)
                    crds[0,s] = np.where(update,self.res*(_ni+ni),crds[0,s])
                    crds[1,s] = np.where(update,self.res*(_ei+ei),crds[1,s])
                    # sets to highest value
                    hangles[s] = np.where(update, h,
                                                  hangles[s])
            elif np.any(np.concatenate([hangles[s] < h
                                        for s in slices])):
                # need to recursively process at higher resolution
                hangles, crds = self.calc_horizon(e0, n0, u0,
                                        n_az=n_az, imp=imp[:-1], f_prev=f,
                                        ei_off=_ei+ei, ni_off=_ni+ni,
                                        crds=crds, hangles=hangles)
            else:
                # can skip this pixel
                pass
        return hangles, crds
