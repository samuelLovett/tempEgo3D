"""
Author:         Samuel Lovett
Date Created:   Apr 26, 2024
Last Edited:    Dec 06, 2024

Description:
RANdom SAmple Consensus program described in Sec. III of [1]

[1] Temporally Constrained Instanteneous Ego-Motion Estimation using 3D Doppler Radar
"""
from tempEgo3D.data_unpacker import data_unpacker_tensor_object_3D
from tempEgo3D.least_squares_3D import least_squares_velocity_analysis_3D
import numpy as np

rng = np.random.default_rng()

class RANSAC:
    """
    Standard ransac class which implements ransac method as shown in Sec. III A. of [1]
    """

    def __init__(self, n=10, k=100, epsilon=0.05, rho=10, loss=None, metric=None):
        """
        Initialization of the standard velocity estimation method (KB) as described at the beginning of this code.
        :param n: Minimum number of samples to estimate parameters
        :param k: Maximum iterations allowed
        :param epsilon: Threshold value to determine if points are fit well
        :param rho: Number of close samples required to assert model fits well
        :param loss: function of `y_true` and `y_pred` that returns a vector scoring how well y_pred matches y_true at each point
        :param metric: function of `y_true` and `y_pred` and returns a float scoring how well y_pred matches y_true overall
        """
        self.n = n
        self.k = k
        self.epsilon = epsilon
        self.rho = rho
        self.loss = loss
        self.metric = metric

    def estimate_velocity(self, buffer):
        """
        Filter the inlier point from the outlier points using RANSAC and return the body velocity. \n
        return vel: list containing the x, y, and z body velocity [velocity in x, velocity in y, velocity in z]
        :param buffer: multidimensional list of measurement data [theta, gamma, doppler] (3, number of samples)
        :return: vel
        """
        measured_azim_val, measured_elev_val, measured_doppler_val, _, _ = buffer
        best_error = np.inf
        index = np.arange(len(measured_azim_val))
        maybe_model = SinusoidalRegressor()
        refined_model = SinusoidalRegressor()
        model_using_all_found_inliers_model = SinusoidalRegressor()

        n = self.n
        epsilon = self.epsilon
        rho = self.rho
        for _ in range(self.k):

            # create the list initial guess inliers by taking the n randomly selected indices
            # and copying the azimuth elevation and doppler values at those locations
            initial_guess_idx = rng.choice(index, n, replace=False, axis=0, shuffle=False)
            guess_inliers_azim = measured_azim_val[initial_guess_idx[:]]
            guess_inliers_elev = measured_elev_val[initial_guess_idx[:]]
            guess_inliers_doppler = measured_doppler_val[initial_guess_idx[:]]

            # make a model with our initial guess of inliers
            maybe_model.fit(guess_inliers_azim, guess_inliers_elev, guess_inliers_doppler)

            # using the created model predict the doppler values at the location of our azimuth values
            model_predicted_doppler_at_given_angles = maybe_model.predict(measured_azim_val, measured_elev_val)

            # create a boolean array that is the same length as our data arrays
            # compare the squared error distance between the predicted value and the measured value.
            # If it is greater than the threshold set that index value to true and if it is lower set the value to false
            thresholded = (self.loss(measured_doppler_val, model_predicted_doppler_at_given_angles) < epsilon)

            # copy the values from the index corresponding to the location of the true values within threshold
            inlier_ids = index[:][np.where(thresholded)[0]]
            number_of_inliers = inlier_ids.size

            # check if the number of inliers from this guess is higher than required to assert model fits well
            if number_of_inliers > rho:
                # fit a model with all the inliers found from the initial guess
                refined_model.fit(measured_azim_val[inlier_ids],
                                  measured_elev_val[inlier_ids],
                                  measured_doppler_val[inlier_ids])

                # for the subset of inliers, calculate the mean squared error between the measured doppler and
                # the predicted doppler from the better model (model using all inlier points)
                this_error = self.metric(measured_doppler_val[inlier_ids],
                                         refined_model.predict(measured_azim_val[inlier_ids],
                                                               measured_elev_val[inlier_ids]))

                # check if the error is less than the best error found so far
                if this_error <= best_error:
                    # update values since this model is better than the last
                    best_error = this_error
                    model_using_all_found_inliers_model = refined_model

        vel = [model_using_all_found_inliers_model.velocity_x,
               model_using_all_found_inliers_model.velocity_y,
               model_using_all_found_inliers_model.velocity_z]

        return vel


class TEMPSAC(RANSAC):
    """
    TEMPSAC class which implements TEMPSAC in Sec. III C. of [1]
    """

    def __init__(self, n=10, k=100, epsilon=0.05, rho=10, loss=None, metric=None):
        super().__init__(n, k, epsilon, rho, loss, metric)

    def get_initial_guess(self, buffer, probability):
        """
        Uses the probabilities to generate the likelihood of selecting a sample from a specific measurement and then uniformly selects samples from the selected measurement.
        The index of the selected samples are then returned for model fitting.
        :param buffer: samples being selected on
        :param probability: weights used to select the samples
        :return: n_random_points_full_idx
        """
        buffer_length = len(buffer)
        buffer_index = np.arange(buffer_length)
        try:
            n_random_buffer_idx = rng.choice(buffer_index, self.n, True, probability, 0, False)
        except Exception as e:
            print(e)
            exit()

        n_random_points_full_idx = []
        buffer_item_bins = []

        # find the length of each element within the buffer
        for m in range(buffer_length):
            number_of_data_points = len(buffer[m][0])
            buffer_item_bins.append(number_of_data_points)

        # Return the number of times each measurement is selected
        unique_buffer_idx, times_selected = np.unique(n_random_buffer_idx, return_counts=True)

        # Select the random samples from within each selected measurement in buffer
        for i in range(len(unique_buffer_idx)):
            num_points_to_select = times_selected[i]
            selected_buffer = unique_buffer_idx[i]
            measurement_idx = np.arange(buffer_item_bins[selected_buffer])
            random_points = rng.choice(measurement_idx, num_points_to_select, replace=False, shuffle=False)

            # Convert the within measurement index into the full index
            if unique_buffer_idx[i] == 0:
                n_random_points_full_idx.extend(random_points)
            else:
                for point_selected in random_points:
                    full_index = sum(buffer_item_bins[:selected_buffer]) + point_selected
                    n_random_points_full_idx.append(full_index)
        return n_random_points_full_idx

    def estimate_velocity(self, buffer):
        """
        Filter the inlier point from the outlier points using TEMPSAC and return the body velocity. \n
        return vel: list containing the x, y, and z body velocity [velocity in x, velocity in y, velocity in z]
        :param buffer: multidimensional list of measurement data [theta, gamma, doppler] (3, number of samples)
        :return: vel
        """
        buffer, probability = buffer
        measured_azim_val, measured_elev_val, measured_doppler_val, = data_unpacker_tensor_object_3D(buffer)
        best_error = np.inf
        epsilon = self.epsilon
        rho = self.rho

        index = np.arange(len(measured_azim_val))
        maybe_model = SinusoidalRegressor()
        better_model = SinusoidalRegressor()
        model_using_all_found_inliers_model = SinusoidalRegressor()

        for _ in range(self.k):

            # if you get an error about this line of code check your n value
            initial_guess_idx = self.get_initial_guess(buffer, probability)

            # create the list initial guess inliers by taking the n randomly selected indices
            # and copying the azimuth and doppler values at those locations
            guess_inliers_azim = measured_azim_val[initial_guess_idx[:]]
            guess_inliers_elev = measured_elev_val[initial_guess_idx[:]]
            guess_inliers_doppler = measured_doppler_val[initial_guess_idx[:]]

            # make a model with our initial guess of inliers
            maybe_model.fit(guess_inliers_azim, guess_inliers_elev, guess_inliers_doppler)

            # using the created model predict the doppler values at the location of our azimuth values
            model_predicted_doppler_at_given_angles = maybe_model.predict(measured_azim_val, measured_elev_val)

            # create a boolean array that is the same length as our data arrays
            # compare the squared error distance between the predicted value and the measured value.
            # If it is greater than the threshold set that index value to true and if it is lower set the value to false
            thresholded = (self.loss(measured_doppler_val, model_predicted_doppler_at_given_angles) < epsilon)

            # copy the values from the index corresponding to the location of the true values within threshold
            inlier_ids = index[:][np.where(thresholded)[0]]
            number_of_inliers = inlier_ids.size

            # check if the number of inliers from this guess is higher than required to assert model fits well
            if number_of_inliers > rho:
                # fit a model with all the inliers found from the initial guess
                better_model.fit(measured_azim_val[inlier_ids],
                                 measured_elev_val[inlier_ids],
                                 measured_doppler_val[inlier_ids])

                # for the subset of inliers, calculate the mean squared error between the measured doppler and
                # the predicted doppler from the better model (model using all inlier points)
                this_error = self.metric(measured_doppler_val[inlier_ids],
                                         better_model.predict(measured_azim_val[inlier_ids],
                                                              measured_elev_val[inlier_ids]))

                # check if the error is less than the best error found so far
                if this_error <= best_error:
                    # update values since this model is better than the last
                    best_error = this_error
                    model_using_all_found_inliers_model = better_model

        vel = [model_using_all_found_inliers_model.velocity_x,
               model_using_all_found_inliers_model.velocity_y,
               model_using_all_found_inliers_model.velocity_z]

        return vel


class RANSAC_TWLSQ(RANSAC):
    """
    TWLSQ ransac class which implements ransac identically to the standard method but passes the weights to the regressor class as shown in Sec. III D. of [1]
    """

    def __init__(self, n=10, k=100, epsilon=0.05, rho=10, loss=None, metric=None):
        super().__init__(n, k, epsilon, rho, loss, metric)

    def estimate_velocity(self, buffer):
        """
        Filter the inlier point from the outlier points using TWLSQ return the body velocity. \n
        return vel: list containing the x, y, and z body velocity [velocity in x, velocity in y, velocity in z]
        :param buffer: multidimensional list of measurement data [theta, gamma, doppler] (3, number of samples)
        :return: vel
        """
        measured_azim_val, measured_elev_val, measured_doppler_val, full_weight_mask = buffer
        best_error = np.inf
        index = np.arange(len(measured_azim_val))
        maybe_model = SinusoidalRegressor()
        better_model = SinusoidalRegressor()
        model_using_all_found_inliers_model = SinusoidalRegressor()

        n = self.n
        epsilon = self.epsilon
        rho = self.rho
        for _ in range(self.k):

            # create the list initial guess inliers by taking the n randomly selected indices
            # and copying the azimuth and doppler values at those locations
            initial_guess_idx = rng.choice(index, n, replace=False, axis=0, shuffle=False)
            guess_inliers_azim = measured_azim_val[initial_guess_idx[:]]
            guess_inliers_elev = measured_elev_val[initial_guess_idx[:]]
            guess_inliers_doppler = measured_doppler_val[initial_guess_idx[:]]

            # make a model with our initial guess of inliers
            weight_mask = full_weight_mask[initial_guess_idx[:]]
            maybe_model.fit(guess_inliers_azim, guess_inliers_elev, guess_inliers_doppler, weights=weight_mask)

            # using the created model predict the doppler values at the location of our azimuth values
            model_predicted_doppler_at_given_angles = maybe_model.predict(measured_azim_val, measured_elev_val)

            # create a boolean array that is the same length as our data arrays
            # compare the squared error distance between the predicted value and the measured value.
            # If it is greater than the threshold set that index value to true and if it is lower set the value to false
            thresholded = (self.loss(measured_doppler_val, model_predicted_doppler_at_given_angles) < epsilon)

            # copy the values from the index corresponding to the location of the true values within threshold
            inlier_ids = index[:][np.where(thresholded)[0]]
            number_of_inliers = inlier_ids.size

            # check if the number of inliers from this guess is higher than required to assert model fits well
            if number_of_inliers > rho:
                # fit a model with all the inliers found from the initial guess
                weight_mask = full_weight_mask[inlier_ids[:]]
                better_model.fit(measured_azim_val[inlier_ids],
                                 measured_elev_val[inlier_ids],
                                 measured_doppler_val[inlier_ids],
                                 weights=weight_mask)

                # for the subset of inliers, calculate the mean squared error between the measured doppler and
                # the predicted doppler from the better model (model using all inlier points)
                this_error = self.metric(measured_doppler_val[inlier_ids],
                                         better_model.predict(measured_azim_val[inlier_ids],
                                                              measured_elev_val[inlier_ids]))

                # check if the error is less than the best error found so far
                if this_error <= best_error:
                    # update values since this model is better than the last
                    best_error = this_error
                    model_using_all_found_inliers_model = better_model


        vel = [model_using_all_found_inliers_model.velocity_x,
               model_using_all_found_inliers_model.velocity_y,
               model_using_all_found_inliers_model.velocity_z]

        return vel



class SinusoidalRegressor:
    """
    Velocity model class which implements (13) from [1].
    """

    def __init__(self):
        self.velocity_x = 0
        self.velocity_y = 0
        self.velocity_z = 0
        self.weights = None

    def fit(self, azimuth, elev, doppler, weights=None):
        """
        Fit the model using LSQ or TWLSQ.
        \n param[in] azimuth: array of azimuth values.
        \n param[in] elev: array of elevation values.
        \n param[in] doppler: array of doppler values.
        \n param[in] weights: Optional array of weights corresponding to each sample.
        \n return self: return the parameterized velocity model object
        :param azimuth:
        :param elev:
        :param doppler:
        :param weights:
        :return: velocity model object
        """
        vx, vy, vz = least_squares_velocity_analysis_3D(azimuth, elev, doppler, weights)
        # Make velocity negative since it is moving in opposite direction of sensor
        # Sentence after equations (1) in "Instantaneous ego-motion estimation using Doppler radar" D. Kellner et al
        self.velocity_x = -1 * vx
        self.velocity_y = -1 * vy
        self.velocity_z = -1 * vz
        return self

    def predict(self, theta: np.ndarray, elev: np.ndarray):
        """
        Predict the doppler velocity from the given azimuth values using (13) from [1]
        :param theta: azimuth values to estimate the doppler velocity for
        :param elev: elevation values to estimate the doppler velocity for
        :return: doppler velocity estimates
        """

        predicted_velocity = -self.velocity_x * np.cos(elev) * np.cos(theta) \
                             + -self.velocity_y * np.cos(elev) * np.sin(theta) \
                             + -self.velocity_z * np.sin(elev)

        return predicted_velocity
