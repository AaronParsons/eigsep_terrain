'''Utility functions for calculating distance and azimuth angles.'''

import numpy as np

R_earth = 6378e3 # m

def az_bin(e, n, n_az):
    '''Calculate azimuthal angle and round to nearest bin.'''
    az = np.arctan2(e[None, :], n[:, None])
    az = np.where(az < 0, 2 * np.pi + az, az)
    b = np.around(az / (2 * np.pi / n_az)).astype(int)
    return b
    
def calc_az_bin_range(e_edges, n_edges, e0, n0, n_az):
    '''Calculate the min/max az ranges a pixel could contain, based on
    where the pixel edges are.'''
    # (0, 0) is bottom-left
    # Letters are axis0: (t=top, m=middle, b=bottom),
    #             axis1: (l=left, c=center, r=right)
    de_edges = e_edges - e0
    dn_edges = n_edges - n0
    if n0 > n_edges[-1]:
        # b case
        b0, b1 = slice(0, -1), slice(1, None)
    elif n0 >= n_edges[0]:
        # full case
        n0_px = np.searchsorted(n_edges, n0) - 1
        b0, b1 = slice(      0, n0_px+0), slice(      1, n0_px+1)
        m0, m1 = slice(n0_px+0, n0_px+1), slice(n0_px+1, n0_px+2)
        t0, t1 = slice(n0_px+1,      -1), slice(n0_px+2,    None)
    else:  # n0_px < n_edges[0]
        # t case
        t0, t1 = slice(0, -1), slice(1, None)
    
    if e0 < e_edges[0]:
        # r case
        r0, r1 = slice(0, -1), slice(1, None)
        if n0 > n_edges[-1]:
            # br case
            ___a = az_bin(de_edges[r1], dn_edges[b1], n_az)
            ___b = az_bin(de_edges[r0], dn_edges[b0], n_az)
        elif n0 >= n_edges[0]:
            # r case
            b__a = az_bin(de_edges[r1], dn_edges[b1], n_az)
            b__b = az_bin(de_edges[r0], dn_edges[b0], n_az)
            m__a = az_bin(de_edges[r0], dn_edges[m1], n_az)
            m__b = az_bin(de_edges[r0], dn_edges[m0], n_az)
            t__a = az_bin(de_edges[r0], dn_edges[t1], n_az)
            t__b = az_bin(de_edges[r1], dn_edges[t0], n_az)
            ___a = np.concatenate([b__a, m__a, t__a], axis=0)
            ___b = np.concatenate([b__b, m__b, t__b], axis=0)
        else:  # n0 < n_edges[0]
            # tr case
            ___a = az_bin(de_edges[r0], dn_edges[t1], n_az)
            ___b = az_bin(de_edges[r1], dn_edges[t0], n_az)
    elif e0 <= e_edges[-1]:
        e0_px = np.searchsorted(e_edges, e0) - 1
        l0, l1 = slice(0, e0_px+0), slice(1, e0_px+1)
        c0, c1 = slice(e0_px+0, e0_px+1), slice(e0_px+1, e0_px+2)
        r0, r1 = slice(e0_px+1, -1), slice(e0_px+2, None)
        if n0 > n_edges[-1]:
            # bc case
            ___a = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l1, c1, r1], [b0, b1, b1])], axis=1)
            ___b = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l0, c0, r0], [b1, b1, b0])], axis=1)
        elif n0 >= n_edges[0]:
            # case
            b__a = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l1, c1, r1], [b0, b1, b1])], axis=1)
            b__b = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l0, c0, r0], [b1, b1, b0])], axis=1)
            m__a = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l1, c1, r0], [m0, m0, m1])], axis=1)
            m__b = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l1, c1, r0], [m1, m1, m0])], axis=1)
            t__a = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l0, c0, r0], [t0, t0, t1])], axis=1)
            t__b = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l1, c1, r1], [t1, t0, t0])], axis=1)
            ___a = np.concatenate([b__a, m__a, t__a], axis=0)
            ___b = np.concatenate([b__b, m__b, t__b], axis=0)
            # manually set middle pixel to full range
            ___a[n0_px, e0_px] = 0
            ___b[n0_px, e0_px] = n_az - 1
        else:  # n0 < n_edges[0]
            # tc case
            ___a = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l0, c0, r0], [t0, t0, t1])], axis=1)
            ___b = np.concatenate([az_bin(de_edges[se], dn_edges[ne], n_az)
                    for se, ne in zip([l1, c1, r1], [t1, t0, t0])], axis=1)
    else:  # e0 > e_edges[-1]
        # l case
        l0, l1 = slice(0, -1), slice(1, None)
        if n0 > n_edges[-1]:
            # bl case
            ___a = az_bin(de_edges[l1], dn_edges[b0], n_az)
            ___b = az_bin(de_edges[l0], dn_edges[b1], n_az)
        elif n0 >= n_edges[0]:
            # l case
            b__a = az_bin(de_edges[l1], dn_edges[b0], n_az)
            b__b = az_bin(de_edges[l0], dn_edges[b1], n_az)
            m__a = az_bin(de_edges[l1], dn_edges[m0], n_az)
            m__b = az_bin(de_edges[l1], dn_edges[m1], n_az)
            t__a = az_bin(de_edges[l0], dn_edges[t0], n_az)
            t__b = az_bin(de_edges[l1], dn_edges[t1], n_az)
            ___a = np.concatenate([b__a, m__a, t__a], axis=0)
            ___b = np.concatenate([b__b, m__b, t__b], axis=0)
        else:  # n0 < n_edges[0]
            # tl case
            ___a = az_bin(de_edges[l0], dn_edges[t0], n_az)
            ___b = az_bin(de_edges[l1], dn_edges[t1], n_az)
            
    return ___a, ___b
    
def calc_rmin(e_edges, n_edges, e0, n0):
    '''Calculate the min r a pixel could contain, based on where the
    pixel edges are.'''
    # (0, 0) is bottom-left
    # Letters are axis0: (t=top, m=middle, b=bottom),
    #             axis1: (l=left, c=center, r=right)
    if n0 <= n_edges[0]:
        # t case
        t = slice(0, -1)
        dn = n_edges[t] - n0
    elif n0 < n_edges[-1]:
        # full case
        n0_px = np.searchsorted(n_edges, n0)
        b = slice(1, n0_px)
        t = slice(n0_px, -1)
        dn = np.concatenate([n_edges[b], np.array([n0]), n_edges[t]]) - n0
    else:  # n0 > n_edges[-1]:
        # b case
        b = slice(1, None)
        dn = n_edges[b] - n0
    
    if e0 <= e_edges[0]:
        # r case
        r = slice(0, -1)
        de = e_edges[r] - e0
    elif e0 < e_edges[-1]:
        e0_px = np.searchsorted(e_edges, e0)
        l = slice(1, e0_px)
        r = slice(e0_px, -1)
        de = np.concatenate([e_edges[l], np.array([e0]), e_edges[r]]) - e0
    else:  # e_edges[-1] < e0_px 
        # l case
        l = slice(1, None)
        de = e_edges[l] - e0
        
    return np.sqrt(dn[:, None]**2 + de[None, :]**2)

def horizon_angle_to_distance(angles, alt):
    '''Given an angle above the horizon (radians) and altitude (m) compute
    a visibility distance (m) accounting for earth curvature.'''
    th3 = np.arcsin(R_earth * np.sin(np.pi/2 + angles) / (R_earth + alt))
    return R_earth * (np.pi/2 - angles - th3)
