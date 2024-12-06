"""
Author:         Samuel Lovett
Date Created:   Apr 26, 2024
Last Edited:    Dec 06, 2024

Description:
Estimate the 3D body-frame velocity vector of the sensor platform given input data from a single radar as described in
[1]. For use with your own radar sensor modify the code in the update_buffer_weighted method.

The velocity estimation scheme takes the following approach:

1. A weighting is applied to all datapoints within the sliding window based on the datapoints corresponding
measurement-index

2. A RANSAC outlier rejection method is used to filter the data which contains a large amount of
outliers. RANSAC selects points uniformly or based on their weights to estimate the velocity. This process is then
repeated until a first-pass velocity estimate is derived from the best inlier set.

3. Least Squares Distance Regression (LSQ) is initialized with the inlier set and used to calculate the final velocity
estimate of the body frame linear velocity.

[1] Temporally Constrained Instanteneous Ego-Motion Estimation using 3D Doppler Radar
"""
from tempEgo3D.error_and_loss_function import *
from tempEgo3D.external_dataset_import import *
from tempEgo3D.ego_motion_methods import *
import numpy as np
from collections import deque
import time


def set_RANSAC(dataset):
    """
    Define the RANSAC method parameter's as described in Sec. III Hyperparameter optimization of [1]
    :param dataset_key: key to use with the dataset directory dictionary
    :return: The velocity estimation object
    """
    n = 3  # Minimum number of data points to estimate parameters
    k = 608  # Maximum iterations allowed
    epsilon = 0.0006194819604855642  # Threshold value to determine if points are fit well
    rho = 31  # Number of points to assert that the model fits well
    start_idx = 0
    stop_idx = None
    method = VelocityEstimation(dataset, start_idx, stop_idx, n, k, epsilon, rho)
    return method


# def set_MLESAC_buffered(dataset):
#     """
#     Define the MLESAC method parameter's as described in
#     :return: The velocity estimation object
#     """
#     m = 2  # Buffer size
#     n = 3  # Minimum number of data points to estimate parameters
#     start_idx = 0
#     stop_idx = None
#     method = MLESACBufferVelEstimate(dataset, start_idx, stop_idx, n, m)
#     return method


def set_TWLSQ(dataset):
    """
    Define the Temporally Weighted Least Squares method parameter's as described in Sec. III Hyperparameter
    optimization of [1]
    :param dataset_key: key to use with the dataset directory dictionary
    :return: The velocity estimation object
    """
    m = 2  # Buffer size
    fff_lambda = 0.15760037633371646  # 0 values only the most recent samples and 1 values all samples equally
    n = 3  # Minimum number of data points to estimate parameters
    k = 68  # Maximum iterations allowed
    epsilon = 14.39032148114001  # Threshold value to determine if points are fit well
    rho = 24  # Number of points to assert that the model fits well
    start_idx = 0
    stop_idx = None

    method = TWLSQVelEstimate(dataset,
                              start_idx,
                              stop_idx,
                              m,
                              fff_lambda,
                              n,
                              k,
                              epsilon,
                              rho)
    return method


def set_TEMPSAC(dataset):
    """
    Define the TEMPoral SAmpling Concensus parameter's as described in Sec. III Hyperparameter
    optimization of [1]
    :param dataset_key: key to use with the dataset directory dictionary
    :return: The velocity estimation object
    """
    m = 2  # Buffer size
    fff_lambda = 0.7160085793075731  # 0 values only the most recent samples and 1 values all samples equally
    n = 3  # Minimum number of data points to estimate parameters
    k = 1494  # Maximum iterations allowed
    epsilon = 14.843480780976819  # Threshold value to determine if points are fit well
    rho = 24  # Number of points to assert that the model fits well
    start_idx = 0
    stop_idx = None

    method = TEMPSACVelEstimate(dataset,
                                start_idx,
                                stop_idx,
                                m,
                                fff_lambda,
                                n,
                                k,
                                epsilon,
                                rho)

    return method


def get_subset_of_weights_normalized(buffer_length, fff_lambda):
    """
    calculates the normalized weight value for each measurement index when the sliding window is not full
    :param buffer_length: length of the sliding window labelled as m in [1]
    :param fff_lambda: fixed factor lambda as defined in (14) of [1]
    :return: list of weight values normalized by the number of measurements in the sliding window
    """

    weights = []
    buffer_length_indexed_from_zero = buffer_length - 1
    for buffer_index in range(buffer_length):
        weight_at_i = fff_lambda ** (buffer_length_indexed_from_zero - buffer_index)
        weights.append(weight_at_i)

    sum_of_weights = sum(weights)
    normalized_weights = [weights[i] / sum_of_weights for i in range(len(weights))]
    return normalized_weights


def get_subset_of_weights(buffer_length, fff_lambda):
    """
    calculates the normalized weight value for each measurement index when the sliding window is not full
    :param buffer_length: length of the sliding window labelled as m in [1]
    :param fff_lambda: fixed factor lambda as defined in (14) of [1]
    :return: list of weight values
    """

    weights = []
    buffer_length_indexed_from_zero = buffer_length - 1
    for buffer_index in range(buffer_length):
        weight_at_i = fff_lambda ** (buffer_length_indexed_from_zero - buffer_index)
        weights.append(weight_at_i)

    return weights


class VelocityEstimation:
    """
    Standard velocity estimation class which implements the RANSAC method as shown in [1]
    """

    def __init__(self, __dataset, __start_idx, __stop_idx, __n, __k, __epsilon, __rho):
        """
        Initialization of the standard velocity estimation method RANSAC as described at the beginning of this code.
        :param __dataset: the dataset directory dictionary
        :param __start_idx: measurement index to start the evaluation at
        :param __stop_idx: measurement index to stop the evaluation at
        :param __n: Minimum number of samples to estimate parameters
        :param __k: Maximum iterations allowed
        :param __epsilon: Threshold value to determine if points are fit well
        :param __rho: Number of close samples required to assert model fits well
        """
        calib_dir = 'coloradar_package//dataset//calib'
        self.coloradar_dataset = ColoradarDataset(__dataset, calib_dir)
        if __stop_idx is None:
            self.stop = self.coloradar_dataset.dataset_length
        else:
            self.stop = __stop_idx
        self.start_idx = __start_idx

        self.velocity_estimator = RANSAC(loss=square_error,
                                                metric=mean_square_error,
                                                n=__n,
                                                k=__k,
                                                epsilon=__epsilon,
                                                rho=__rho)

    def update_buffer(self, idx):
        """
        Update the sliding window as described in Sec. III B. of [1]. Index m is the most recent measurement in time and index 0 is the oldest measurement in time. \n
        \n param[in] idx: measurement index to import from coloradar
        \n return tensor_stack: multidimensional array dimension 1 is [theta, gamma, vd], and dimension 2 is the sample index
        \n return weight_mask: a list of weight values corresponding to each sample in tensor_stack.\n
        For use with your own data replace the line: current_sample = self.coloradar_dataset.get_radar_cloud(idx) and data must be formatted as a 3D list [theta, gamma, doppler].
        \t theta is a list of azimuth values in rads. \n
        \t gamma is a list of elevation values in rads. \n
        \t doppler is a list of doppler velocity values in m/s.

        :param idx: current measurement index used to import values from the dataset
        :return: current_sample
        """

        # Get the new measurement
        current_sample = self.coloradar_dataset.get_radar_cloud_3D(idx)
        return current_sample


# class MLESACBufferVelEstimate(VelocityEstimation):
#     """
#     MLESAC Buffered velocity estimation class which inherits the initialization of the standard method but uses MLESAC from
#     goggles.
#     """
#
#     def __init__(self, __dataset, __start_idx, __stop_idx, __buffer_size, __n):
#         super().__init__(__dataset, __start_idx, __stop_idx, __n, None, None, None)
#         if __dataset_key is not None:
#             __results_location = 'results//buffer//' + __dataset_key + '//mlesacBuffered//'
#             self.my_results = MLESACBufferedResults(__results_location)
#         self.my_opt_results = OptResultsClass()
#
#         self.velocity_estimator = MLESACEstimator()
#         self.buffer_size = __buffer_size
#         self.sliding_window = []
#
#     def update_buffer(self, idx):
#         """
#         Update the sliding window as described in Sec. II C. of [1]. Index m is the most recent measurement in time and index 0 is the oldest measurement in time. \n
#         \n param[in] idx: measurement index to import from coloradar
#         \n return current measurement: multidimensional array dimension 1 is [theta, vd], and dimension 2 is the sample index
#         For use with your own data replace the line: current_sample = self.coloradar_dataset.get_radar_cloud(idx) and data must be formatted as a 2D list [theta, doppler].
#         \t theta is a list of azimuth values in rads. \n
#         \t doppler is a list of doppler velocity values in m/s.
#
#         :param idx: current measurement index used to import values from the dataset
#         :return: current measurement
#         """
#         # Get the new measurement
#         current_sample = self.coloradar_dataset.get_radar_cloud_3D(idx)
#         buffer_length = len(self.sliding_window)
#
#         current_buffer = self.sliding_window
#
#         # append the current measurement to the buffer
#         if buffer_length < self.buffer_size:
#             current_buffer.append(current_sample)
#             buffer_length = buffer_length + 1
#         else:
#             # Remove the oldest measurement from the list
#             # Update the buffer to include the new measurement
#             current_buffer.pop(0)
#             current_buffer.append(current_sample)
#
#         # Assemble the multidimensional array
#         reshaped_tensor_stack = np.concatenate(list(current_buffer), axis=1)
#         theta = reshaped_tensor_stack[0, :]
#         gamma = reshaped_tensor_stack[1, :]
#         doppler = reshaped_tensor_stack[2, :]
#         object_range = reshaped_tensor_stack[3, :]
#         intensity = reshaped_tensor_stack[4, :]
#         return theta, gamma, doppler, object_range, intensity


class TWLSQVelEstimate(VelocityEstimation):
    """
    TWLSQ velocity estimation class which inherits the initialization of the standard method but uses TWLSQ_RANSAC
    as shown in [1].
    """

    def __init__(self, __dataset, __start_idx, __stop_idx, __buffer_size,
                 __fff_lambda, __n, __k, __epsilon, __rho):
        """
        Initialization of the velocity estimation method as described at the beginning of this code.
        :param __dataset_key: key to use with the dataset directory dictionary
        :param __start_idx: measurement index to start the evaluation at
        :param __stop_idx: measurement index to stop the evaluation at
        :param __buffer_size: Size of the sliding window, m in [1]
        :param __fff_lambda: fixed forgetting factor as defined in (14) of [1]. 0 uses only the most recent samples and 1 uses all samples equally.
        :param __n: Minimum number of samples to estimate parameters
        :param __k: Maximum iterations allowed
        :param __epsilon: Threshold value to determine if points are fit well
        :param __rho: Number of close samples required to assert model fits well
        """
        super().__init__(__dataset, __start_idx, __stop_idx, __n, __k, __epsilon, __rho)

        self.velocity_estimator = RANSAC_TWLSQ(loss=square_error,
                                                      metric=mean_square_error,
                                                      n=__n,
                                                      k=__k,
                                                      epsilon=__epsilon,
                                                      rho=__rho)

        fff_lambda = __fff_lambda

        weights = []
        buffer_size_indexed_from_zero = __buffer_size - 1
        for buffer_index in range(__buffer_size):
            weight_at_i = fff_lambda ** (buffer_size_indexed_from_zero - buffer_index)
            weights.append(weight_at_i)

        self.exponential_weights_list = weights

        self.buffer_size = __buffer_size
        self.sliding_window = []
        self.fff_lambda = fff_lambda

    def update_buffer(self, idx):
        """
        Update the sliding window as described in Sec. III B. of [1]. Index m is the most recent measurement in time and index 0 is the oldest measurement in time. \n
        \n param[in] idx: measurement index to import from coloradar
        \n return tensor_stack: multidimensional array dimension 1 is [theta, gamma, vd], and dimension 2 is the sample index
        \n return weight_mask: a list of weight values corresponding to each sample in tensor_stack.\n
        For use with your own data replace the line: current_sample = self.coloradar_dataset.get_radar_cloud(idx) and data must be formatted as a 2D list [theta, doppler].
        \t theta is a list of azimuth values in rads. \n
        \t gamma is a list of elevation values in rads. \n
        \t doppler is a list of doppler velocity values in m/s.

        :param idx: current measurement index used to import values from the dataset
        :return: tensor_stack, weight_mask
        """

        # Get the new measurement
        current_sample = self.coloradar_dataset.get_radar_cloud_3D(idx)

        if current_sample is None:
            return None

        buffer_length = len(self.sliding_window)

        current_buffer = self.sliding_window

        # append the current measurement to the buffer
        if buffer_length < self.buffer_size:
            current_buffer.append(current_sample)
            buffer_length = buffer_length + 1
            # weights = get_subset_of_weights_normalized(buffer_length, self.fff_lambda)
            weights = get_subset_of_weights(buffer_length, self.fff_lambda)
        else:
            # Remove the oldest measurement from the list
            # Update the buffer to include the new measurement
            current_buffer.pop(0)
            current_buffer.append(current_sample)
            # weights = self.normalized_exponential_weights_list
            weights = self.exponential_weights_list

        # update the corresponding weight for each sample within the buffer depending on its measurement index
        weight_mask = np.array([])
        for i in range(buffer_length):
            mask_at_i = np.ones(len(current_buffer[i][0])) * weights[i]
            weight_mask = np.concatenate((weight_mask, mask_at_i))

        # Assemble the multidimensional array
        reshaped_tensor_stack = np.concatenate(list(current_buffer), axis=1)
        return reshaped_tensor_stack[0, :], reshaped_tensor_stack[1, :], reshaped_tensor_stack[2,
                                                                         :], weight_mask  # break it into theta, vd, and w


class TEMPSACVelEstimate(VelocityEstimation):
    """
    TEMPSAC velocity estimation class which inherits the initialization of the standard method and the TWLSQ method but
    uses TEMPSAC as shown in [1].
    """

    def __init__(self, __dataset, __start_idx, __stop_idx, __buffer_size,
                 __fff_lambda, __n, __k, __epsilon, __rho):
        """
        Initialization of the velocity estimation method as described at the beginning of this code.
        :param __dataset_key: key to use with the dataset directory dictionary
        :param __start_idx: measurement index to start the evaluation at
        :param __stop_idx: measurement index to stop the evaluation at
        :param __buffer_size: Size of the sliding window, m in [1]
        :param __fff_lambda: fixed forgetting factor as defined in (14) of [1]. 0 uses only the most recent samples and 1 uses all samples equally.
        :param __n: Minimum number of samples to estimate parameters
        :param __k: Maximum iterations allowed
        :param __epsilon: Threshold value to determine if points are fit well
        :param __rho: Number of close samples required to assert model fits well
        """
        super().__init__(__dataset,
                         __start_idx,
                         __stop_idx,
                         __n,
                         __k,
                         __epsilon,
                         __rho)
        self.velocity_estimator = TEMPSAC(loss=square_error,
                                                 metric=mean_square_error,
                                                 n=__n,
                                                 k=__k,
                                                 epsilon=__epsilon,
                                                 rho=__rho)

        fff_lambda = __fff_lambda


        weights = []
        buffer_size_indexed_from_zero = __buffer_size - 1
        for buffer_index in range(__buffer_size):
            weight_at_i = fff_lambda ** (buffer_size_indexed_from_zero - buffer_index)
            weights.append(weight_at_i)
        sum_of_weights = sum(weights)
        normalized_weights = [weights[i] / sum_of_weights for i in range(len(weights))]

        self.normalized_exponential_weights_list = normalized_weights

        self.buffer_size = __buffer_size
        self.sliding_window = []
        self.fff_lambda = fff_lambda

    def update_buffer(self, idx):
        """
        Update the sliding window as described in Sec. III B. of [1]. Index m is the most recent measurement in time and index 0 is the oldest measurement in time. \n
        \n param[in] idx: measurement index to import from coloradar
        \n return tensor_stack: multidimensional array dimension 1 is [theta, gamma, vd], and dimension 2 is the sample index
        \n return weight_mask: a list of weight values corresponding to each sample in tensor_stack.\n
        For use with your own data replace the line: current_sample = self.coloradar_dataset.get_radar_cloud(idx) and data must be formatted as a 2D list [theta, doppler].
        \t theta is a list of azimuth values in rads. \n
        \t gamma is a list of elevation values in rads. \n
        \t doppler is a list of doppler velocity values in m/s.

        :param idx: current measurement index used to import values from the dataset
        :return: tensor_stack, weights
        """

        current_sample = self.coloradar_dataset.get_radar_cloud_3D(idx)  # Get the new item from the sensor

        buffer_length = len(self.sliding_window)
        current_buffer = self.sliding_window

        if buffer_length < self.buffer_size:
            current_buffer.append(current_sample)
            buffer_length = buffer_length + 1
            weights = get_subset_of_weights_normalized(buffer_length, self.fff_lambda)
        else:
            current_buffer.pop(0)  # Remove the oldest item from the list
            current_buffer.append(current_sample)  # Update the buffer to include the new item
            weights = self.normalized_exponential_weights_list

        weight_mask = np.array([])
        for i in range(buffer_length):
            mask_at_i = np.ones(len(current_buffer[i][0])) * weights[i]
            weight_mask = np.concatenate((weight_mask, mask_at_i))

        return current_buffer, weights



def main():
    """
    Example main function. \n
    Modify the paths contained in datasets_dic and calib_dir defined in the class VelocityEstimation __init__ function if using a different directory structure than described in the readme.
    """

    datasets_dic = {'army_run0': 'coloradar_package/dataset/2_23_2021_edgar_army_run0/',
                    'army_run1': 'coloradar_package/dataset/2_23_2021_edgar_army_run1/',
                    'army_run2': 'coloradar_package/dataset/2_23_2021_edgar_army_run2/',
                    'army_run3': 'coloradar_package/dataset/2_23_2021_edgar_army_run3/',
                    'army_run4': 'coloradar_package/dataset/2_23_2021_edgar_army_run4/',
                    'classroom_run2': 'coloradar_package/dataset/2_23_2021_edgar_classroom_run2/',
                    'classroom_run3': 'coloradar_package/dataset/2_23_2021_edgar_classroom_run3/',
                    'classroom_run4': 'coloradar_package/dataset/2_23_2021_edgar_classroom_run4/',
                    'classroom_run5': 'coloradar_package/dataset/2_23_2021_edgar_classroom_run5/',
                    'hallways_run0': 'coloradar_package/dataset/12_21_2020_ec_hallways_run0/',
                    'hallways_run1': 'coloradar_package/dataset/12_21_2020_ec_hallways_run1/',
                    'hallways_run2': 'coloradar_package/dataset/12_21_2020_ec_hallways_run2/',
                    'hallways_run3': 'coloradar_package/dataset/12_21_2020_ec_hallways_run3/',
                    'irl_run0': 'coloradar_package/dataset/12_21_2020_irl_lab_run0/',
                    'irl_run1': 'coloradar_package/dataset/12_21_2020_irl_lab_run1/',
                    'irl_run3': 'coloradar_package/dataset/12_21_2020_irl_lab_run3/',
                    'irl_run4': 'coloradar_package/dataset/12_21_2020_irl_lab_run4/'
                    }

    key = 'classroom_run2'
    dataset = datasets_dic[key]

    start_time = time.time()

    standard = set_RANSAC(dataset)
    TWLSQ = set_TWLSQ(dataset)
    TEMPSAC = set_TEMPSAC(dataset)

    """
    Enter your code here. The following loop calculates the ego velocity of the platform.
    """
    ransac = standard.velocity_estimator
    idx = standard.start_idx
    stop = standard.stop

    while idx < stop:
        radar_cloud = standard.update_buffer(idx)
        velocities_at_idx = ransac.estimate_velocity(radar_cloud)
        print(velocities_at_idx)
        idx = idx + 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == '__main__':
    main()
