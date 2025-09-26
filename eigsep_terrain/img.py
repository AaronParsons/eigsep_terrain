import os
import numpy as np
from matplotlib.image import imread
import cv2
from .utils import rot_m
from .ray import ray_trace_basic
from transformers import pipeline

PRM_ORDER = ('e', 'n', 'u', 'th', 'ph', 'ti', 'f')

def pixels_to_rays(Nu, Nv, f, uv=None, dtype=np.float32):
    if uv is None:
        _u = np.arange(Nu, dtype=int)
        _v = np.arange(Nv, dtype=int)
        uv = np.meshgrid(_v, _u)[::-1]
    u, v = uv
    rays = np.array([Nu // 2 - u, Nv // 2 - v, np.full(u.shape, f)], dtype=dtype)
    rays /= np.linalg.norm(rays, axis=0)
    return rays

class HorizonImage:
    def __init__(self, filename, meta={}, **kwargs):
        self.filename = filename
        self.key = os.path.basename(filename).split('_')[-1].split('.')[0]
        self.npzfile = 'img_seg_' + os.path.basename(filename).replace('jpg','npz')
        self.img = np.flipud(imread(self.filename))
        
        if not os.path.exists(self.npzfile):
            segdict = self.segment_image()
            self.save_segment_image(segdict)
        self.sky_mask = self.read_skymask()
        self.horizon_mask = self.mask_near_horizon()
        
        if self.key in meta:
            self.meta = meta[self.key]
            self.set_prms(self.meta['best_prms'])
        else:
            self.set_prms([0.0 for k in PRM_ORDER])
        self.reset_pixel_choice()

    def set_prms(self, prms):
        self.prms = dict(zip(PRM_ORDER, prms))

    def get_prms(self):
        return (self.prms[k] for k in PRM_ORDER)

    @property
    def prms_str(self):
        return f"{self.prms['e']: 7.2f}, {self.prms['n']: 7.2f},  {self.prms['u']: 7.2f}, {self.prms['th']: 6.4f}, {self.prms['ph']: 6.4f}, {self.prms['ti']: 5.4f}, {self.prms['f']: 7.2f}"

    @property
    def npix_y(self):
        return self.img.shape[0]
        
    @property
    def npix_x(self):
        return self.img.shape[1]
        
    def segment_image(self):
        segformer = pipeline(task="image-segmentation",
                             model="nvidia/segformer-b0-finetuned-ade-512-512",
                             dtype=torch.float16)
        seg = segformer(self.filename)
        return {d['label']: d['mask'] for d in seg}

    def save_segment_image(self, segdict):
        np.savez(self.npzfile, **segdict)

    def read_skymask(self, fill_thresh=200**2):
        npz = np.load(self.npzfile)
        skymask = npz['sky']
        # remove segmenting holes from presence of antenna
        num_labels, labels, stats, cens = cv2.connectedComponentsWithStats((skymask == 0).astype(np.uint8), connectivity=8)
        for label in range(1, num_labels):   # skip background (label 0)
            area = stats[label, cv2.CC_STAT_AREA]
            if area < fill_thresh:
                skymask[labels == label] = 255
        skymask = np.flipud((skymask == 255).astype(bool))
        return skymask
        
    def get_rays(self, pixels=None, dtype=np.float32):
        z_rays = pixels_to_rays(self.npix_y, self.npix_x,
                                f=self.prms['f'], uv=pixels, dtype=dtype)
        rm_tilt = rot_m(self.prms['ti'], np.array([0,0,1], dtype=dtype))
        rm_th   = rot_m(self.prms['th'], np.array([0,1,0], dtype=dtype))
        rm_ph   = rot_m(self.prms['ph'], np.array([0,0,1], dtype=dtype))
        rm = rm_ph @ (rm_th @ rm_tilt)
        rays = np.einsum('ij,j...->i...', rm, z_rays)
        return rays

    def mask_near_horizon(self, px_dist=150):
        m = (self.sky_mask > 0).astype(np.uint8) * 255
        kernel = np.ones((3,3), np.uint8)
        inner_eroded = cv2.erode(m, kernel, iterations=1)
        edge = cv2.bitwise_xor(m, inner_eroded)
        inv_edge = cv2.bitwise_not(edge)
        dist = cv2.distanceTransform(inv_edge, cv2.DIST_L2, 5)
        near_horizon = (dist <= px_dist)
        return near_horizon

    def choose_pixels(self, N=1000, mask=None):
        if mask is None:
            mask = self.horizon_mask
        x, y = np.where(mask)
        inds = np.random.choice(np.arange(x.size), size=N)
        return (x[inds], y[inds])

    def ray_distance(self, dem, rays, dtype=np.float32):
        rays_2d = rays.reshape(rays.shape[0], -1) 
        (E, N), U = dem.get_en(), dem.data
        start_point = np.array([self.prms[k] for k in 'enu'], dtype=dtype)
        delta_r_prev = None
        #for delta_r_m in 5**np.arange(2, -1, -1):
        for delta_r_m in 5**np.arange(1, -1, -1):
            if delta_r_prev is None:
                r = ray_trace_basic(E, N, U, start_point, rays_2d,
                                           delta_r_m=delta_r_m)
            else:
                r_a = ray_trace_basic(E, N, U, start_point, rays_2d,
                        max_iter=int(2 * delta_r_prev / delta_r_m),
                        r_start=(r-delta_r_prev).clip(delta_r_m),
                        delta_r_m=delta_r_m)
                r_b = ray_trace_basic(E, N, U, start_point, rays_2d,
                        r_max=delta_r_prev, delta_r_m=delta_r_m)
                r = np.where(np.isnan(r_b), r_a, r_b)
            delta_r_prev = delta_r_m
        r.shape = rays.shape[1:]
        return r

    def reset_pixel_choice(self):
        self._px_choice = None
        self._best_loss = np.inf
        
    def horizon_ray_loss(self, dem, cnt=1000, dtype=np.float32, verbose=True):
        if self._px_choice is None:
            self._px_choice = self.choose_pixels(N=cnt)
        x_px, y_px = self._px_choice
        is_sky = self.sky_mask[x_px, y_px]
        rays = self.get_rays(pixels=(x_px, y_px), dtype=dtype)
        r = self.ray_distance(dem, rays, dtype=dtype)
        model_sky = np.isnan(r)
        loss = np.mean(is_sky != model_sky)
        if verbose and loss < self._best_loss:
            self._best_loss = loss
            print(f"'best_prms': ({self.prms_str}),  #[LOSS={loss: 6.4f}]", flush=True)
        return loss
