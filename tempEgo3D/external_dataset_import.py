"""
Author:         Samuel Lovett
Date Created:   Apr 26, 2024
Last Edited:    Dec 06, 2024

Description:
Coloaradar Data importing script. Acts as an API between the coloradar dataset and the code in this repository

[1] Temporally Constrained Instanteneous Ego-Motion Estimation using 3D Doppler Radar
[2] Kramer A, Harlow K, Williams C, Heckman C. ColoRadar: The direct 3D millimeter wave radar dataset.
      The International Journal of Robotics Research. 2022;41(4):351-360. doi:10.1177/02783649211068535
"""

from tempEgo3D.dataset_loaders import *
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tempEgo3D.radar_utilities import AIRE_filtering


class ColoradarDataset:
    """
    Coloaradar Data importing class. Acts as an API between the coloradar dataset and the code in this repository
    """
    def __init__(self, dataset_dir, calibration_directory='..//coloradar_package//dataset//calib', __n=3):
        self.dataset_dir = dataset_dir
        calibration_dir = calibration_directory

        all_radar_params = get_single_chip_params(calibration_dir)
        radar_params = all_radar_params['pointcloud']
        gt_params = get_groundtruth_params()

        # make transform from base frame to sensor frame as 4x4 transformation matrix
        H_bs = np.eye(4)
        H_bs[:3, 3] = radar_params['translation']
        H_bs[:3, :3] = Rotation.from_quat(radar_params['rotation']).as_matrix()
        self.H_bs = H_bs

        base_to_sensor_rot = np.array([[0, -1, 0],
                                       [1, 0, 0],
                                       [0, 0, 1]])
        self.base_to_sensor_rot_mat = base_to_sensor_rot

        radar_timestamps = get_timestamps(dataset_dir, radar_params)
        gt_timestamps = get_timestamps(dataset_dir, gt_params)
        gt_poses_unaligned = get_groundtruth(dataset_dir)

        self.gt_poses, self.radar_indices = interpolate_poses(gt_poses_unaligned, gt_timestamps, radar_timestamps)

        self.initial_time = radar_timestamps[self.radar_indices[0]]
        self.dataset_length = len(self.radar_indices)

        self.radar_params = radar_params
        self.radar_timestamps = radar_timestamps

        # set thresholds
        self.n = __n
        self.azimuth_thres_ = 80
        self.elevation_thres_ = 80
        self.range_thres_ = 0.3
        self.range_thres_max_ = 12
        self.intensity_thres_ = 1
        self.doppler_max_ = 2.56
        self.thresholds_ = np.array([self.azimuth_thres_, self.intensity_thres_,
                                     self.range_thres_, self.range_thres_max_,
                                     self.elevation_thres_, self.doppler_max_])

    def get_radar_cloud_3D(self, idx):
        """
        Method modified from COLORADAR package data importer [2]. \n
        Imports the radar cloud from the coloradar dataset and filters it using the AIRE filter.\n
        param[in] idx: measurement index to grab \n
        return radar_cloud: 5D list of filtered radar sample azimuth, elevation, doppler, range, and intensity values [filtered_theta, filtered_gamma, filtered_doppler, filtered_range, filtered_intensity] \n
        :param idx: measurement index to grab
        :return: radar_cloud
        """
        # Skips the first _ # of messages where there is no ground truth value and then uses the remaining messages
        # for hallways_run0 it skips the first 50
        matched_idx = self.radar_indices[idx]
        radar_pc_local = get_pointcloud(matched_idx, self.dataset_dir, self.radar_params)

        x = radar_pc_local[:, 0]
        y = radar_pc_local[:, 1]
        z = radar_pc_local[:, 2]
        intensity = radar_pc_local[:, 3]
        doppler_array = radar_pc_local[:, 4]
        range_val = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x) # this is equivalent to atan(y/x) since x is always greater than 0
        xy = np.sqrt(x ** 2 + y ** 2)
        gamma = np.arctan2(z, xy)


        ## apply AIRE thresholding to remove near-field targets
        ## [azimuth, intensity, range, elevation, doppler]
        data_AIRE = np.column_stack((theta, intensity, range_val, gamma, doppler_array))

        idx_AIRE, Ntargets_valid = AIRE_filtering(data_AIRE, self.thresholds_)
        # print(Ntargets_valid)

        if Ntargets_valid < self.n:
            # print('Not enough valid points to estimate the velocity in TWLSQ')
            return None

        ## define pre-filtered radar data for further processing
        filtered_theta = theta[idx_AIRE]
        filtered_gamma = gamma[idx_AIRE]
        filtered_doppler = doppler_array[idx_AIRE]
        filtered_range = range_val[idx_AIRE]
        filtered_intensity = intensity[idx_AIRE]

        radar_cloud = [filtered_theta, filtered_gamma, filtered_doppler, filtered_range, filtered_intensity]

        return radar_cloud


    def get_gt_at(self, idx):
        """
        Get the ground truth translation and orientation for a given sample index.

        return translation: translation vector [x, y, z] in meters \n
        return orientation: 3x3 rotation matrix corresponding to the platforms orientation
        :param idx: sample index
        :return: translation, rotation
        """
        # GT location given as a homogeneous matrix with reference to the world frame
        orientation = self.gt_poses[idx][:3, :3]

        # Split into rotation and translation
        translation = self.gt_poses[idx][:3, 3]

        return translation, orientation

    def get_radar_time(self, idx):
        radar_timestamp_at_idx = self.radar_timestamps[self.radar_indices[idx]] - self.initial_time
        return radar_timestamp_at_idx

    def get_base_to_sensor_transformation_matrix(self):
        return self.base_to_sensor_rot_mat


def interpolate_poses(src_poses, src_stamps, tgt_stamps):
    """
    This code was written by the authors of [2] and is available at https://arpg.github.io/coloradar/ \n
    As described in [2]: Finds the ground truth pose of the platform at the radar timestamp.\n
    param[in] src_poses: list of poses in the form of a dict {'position': [x,y,z], 'orientation: [x,y,z,w]'} \n
    param[in] src_stamps: list of timestamps for the src_poses \n
    pram[in] tgt_stamps: list of timestamps for which poses need to be calculated \n
    return tgt_poses: list of interpolated poses as 4x4 transformation matrices \n
    return tgt_indices: list of indices in tgt_stamps for which poses were able to be interpolated \n
    :param src_poses: src_poses: list of poses in the form of a dict {'position': [x,y,z], 'orientation: [x,y,z,w]'}
    :param src_stamps: src_stamps: list of timestamps for the src_poses
    :param tgt_stamps: tgt_stamps: list of timestamps for which poses need to be calculated
    :return: tgt_poses, tgt_indices
    """
    src_start_idx = 0
    tgt_start_idx = 0
    src_end_idx = len(src_stamps) - 1
    tgt_end_idx = len(tgt_stamps) - 1

    # ensure first source timestamp is immediately before first target timestamp
    while tgt_start_idx < tgt_end_idx and tgt_stamps[tgt_start_idx] < src_stamps[src_start_idx]:
        tgt_start_idx += 1

    # ensure last source timestamp is immediately after last target timestamp
    while tgt_end_idx > tgt_start_idx and tgt_stamps[tgt_end_idx] > src_stamps[src_end_idx]:
        tgt_end_idx -= 1

    # iterate through target timestamps,
    # interpolating a pose for each as a 4x4 transformation matrix
    tgt_idx = tgt_start_idx
    src_idx = src_start_idx
    tgt_poses = []
    while tgt_idx <= tgt_end_idx and src_idx <= src_end_idx:
        # find source timestamps bracketing target timestamp
        while src_idx + 1 <= src_end_idx and src_stamps[src_idx + 1] < tgt_stamps[tgt_idx]:
            src_idx += 1

        # get interpolation coefficient
        c = ((tgt_stamps[tgt_idx] - src_stamps[src_idx])
             / (src_stamps[src_idx + 1] - src_stamps[src_idx]))

        # interpolate position
        pose = np.eye(4)
        pose[:3, 3] = ((1.0 - c) * src_poses[src_idx]['position']
                       + c * src_poses[src_idx + 1]['position'])

        # interpolate orientation
        r_src = Rotation.from_quat([src_poses[src_idx]['orientation'],
                                    src_poses[src_idx + 1]['orientation']])
        slerp = Slerp([0, 1], r_src)
        pose[:3, :3] = slerp([c])[0].as_matrix()

        tgt_poses.append(pose)

        # advance target index
        tgt_idx += 1

    tgt_indices = range(tgt_start_idx, tgt_end_idx + 1)

    return tgt_poses, tgt_indices