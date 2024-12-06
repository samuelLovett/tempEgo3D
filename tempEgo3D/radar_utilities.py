"""
This code has been developed by Carl Stahoviak and then modified by Samuel Lovett

Original code available at https://github.com/cstahoviak/goggles
"""

from __future__ import division
import numpy as np
from functools import reduce



## filtering of 2D and 3D radar data
def AIRE_filtering(data_AIRE, thresholds):
    ## unpack data (into row vectors)
    radar_azimuth = data_AIRE[:, 0]  # [rad]
    radar_intensity = data_AIRE[:, 1]  # [dB]
    radar_range = data_AIRE[:, 2]  # [m]
    radar_elevation = data_AIRE[:, 3]  # [rad]
    radar_doppler = data_AIRE[:, 3]  # [m/s]

    azimuth_thres = thresholds[0]  # [deg]
    intensity_thres = thresholds[1]  # [dB]
    range_thres = thresholds[2]  # [m]
    range_thres_max = thresholds[3]  # [m]
    elevation_thres = thresholds[4]  # [deg]
    doppler_thres = thresholds[5]  # [m/s]

    ## Indexing in Python example
    ## print("Values bigger than 10 =", x[x>10])
    ## print("Their indices are ", np.nonzero(x > 10))
    idx_azimuth = np.nonzero(np.abs(np.rad2deg(radar_azimuth)) < azimuth_thres)
    idx_intensity = np.nonzero(radar_intensity > intensity_thres)
    idx_range = np.nonzero(radar_range > range_thres)
    idx_range_max = np.nonzero(radar_range < range_thres_max)
    idx_doppler = np.nonzero(np.abs(radar_doppler < doppler_thres))

    ## 3D radar data
    idx_elevation = np.nonzero(np.abs(np.rad2deg(radar_elevation)) < elevation_thres)
    idx_AIRE = reduce(np.intersect1d, (idx_azimuth, idx_intensity, idx_range,
                                       idx_range_max, idx_elevation, idx_doppler))
    Ntargets_valid = idx_AIRE.shape[0]

    return idx_AIRE, Ntargets_valid