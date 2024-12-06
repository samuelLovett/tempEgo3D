"""
Author:         Samuel Lovett
Date Created:   Apr 26, 2024
Last Edited:    Dec 06, 2024

Description:
Flexible script which performs Least-Squares and Temporally Weighted Least Squares regression depending on the arguments provided.
Program described executes the logic described in Sec. III of [1]

[1] Temporally Constrained Instanteneous Ego-Motion Estimation using 3D Doppler Radar
"""

import numpy as np


def build_A_matrix(theta_val, gamma_val):
    """
    Assemble the azimuth values into the A matrix in
    :param theta_val: theta values to build into an matrix
    :param gamma_val: gamma values to build into an matrix
    :return: A matrix used in the LSQ or TWLSQ solution to Ax=b
    """
    A_mat_col_0 = (np.cos(gamma_val) * np.cos(theta_val)).reshape(-1, 1)
    A_mat_col_1 = (np.cos(gamma_val) * np.sin(theta_val)).reshape(-1, 1)
    A_mat_col_3 = np.sin(gamma_val).reshape(-1, 1)
    A_mat = np.concatenate((A_mat_col_0, A_mat_col_1, A_mat_col_3), axis=1)
    return A_mat


def least_squares_velocity_analysis_3D(theta_val, gamma_val, doppler, weights=None):
    """
    Least-Squares and Temporally Weighted Least Squares regression depending on the arguments provided.

    :param theta_val: array of theta values
    :param gamma_val: array of theta values
    :param doppler: array of doppler values
    :param weights: array of weights corresponding to each theta value
    :return: Vx, Vy, Velocity, angle_rad
    """
    A = build_A_matrix(theta_val, gamma_val)
    b = doppler

    # calculate the weighted A and b matrix to perform W-LSQ
    if weights is not None:
        WA = weights[:, np.newaxis]
        A = A * np.sqrt(WA)
        b = b * np.sqrt(weights)

    Vx, Vy, Vz = np.linalg.lstsq(A, b, rcond=None)[0]

    return Vx, Vy, Vz
