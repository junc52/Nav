#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the authors' names nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import numpy as np
import math
from Simulation.utils.frame_transform import *


def gnss_ls_position_velocity(GNSS_measurements, no_GNSS_meas, predicted_r_ea_e, predicted_v_ea_e):
    """
    GNSS_LS_position_velocity - Calculates position, velocity, clock offset,
    and clock drift using unweighted iterated least squares. Separate
    calculations are implemented for position and clock offset and for
    velocity and clock drift

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves

    Inputs:
      GNSS_measurements     GNSS measurement data:
        Column 1              Pseudo-range measurements (m)
        Column 2              Pseudo-range rate measurements (m/s)
        Columns 3-5           Satellite ECEF position (m)
        Columns 6-8           Satellite ECEF velocity (m/s)
      no_GNSS_meas          Number of satellites for which measurements are
                            supplied
      predicted_r_ea_e      prior predicted ECEF user position (m)
      predicted_v_ea_e      prior predicted ECEF user velocity (m/s)

    Outputs:
      est_r_ea_e            estimated ECEF user position (m)
      est_v_ea_e            estimated ECEF user velocity (m/s)
      est_clock             estimated receiver clock offset (m) and drift (m/s)

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """

    # CONSTANTS
    c = 299792458            # Speed of light in m / s
    omega_ie = 7.292115E-5   # Earth rotation rate in rad / s
    DEG_TO_RAD = 0.01745329252

    # Position and clock offset

    # Setup predicted state
    x_pred = np.zeros((4, 1))
    x_pred[0:3, [0]] = predicted_r_ea_e
    test_convergence = 1

    # Repeat until convergence
    pred_meas = np.zeros((no_GNSS_meas, 1))
    H_matrix = np.zeros((no_GNSS_meas, 4))
    while test_convergence > 0.0001:

        # Loop measurements
        for j in range(0, no_GNSS_meas):
            # Predict approximate range
            delta_r = np.transpose(GNSS_measurements[[j], 2:5]) - x_pred[0:3, [0]]
            approx_range = math.sqrt(np.matmul(np.transpose(delta_r), delta_r))

            # Calculate frame rotation during signal transit time using (8.36)
            C_e_I = np.zeros((3, 3))
            C_e_I[0, 0] = 1
            C_e_I[0, 1] = omega_ie * approx_range / c
            C_e_I[1, 0] = -omega_ie * approx_range / c
            C_e_I[1, 1] = 1
            C_e_I[2, 2] = 1

            # Predict pseudo-range using (9.143) (8.35)
            delta_r = np.matmul(C_e_I, np.transpose(GNSS_measurements[[j], 2:5])) - x_pred[0:3, [0]]
            range_ = math.sqrt(np.matmul(np.transpose(delta_r), delta_r))
            pred_meas[j, 0] = range_ + x_pred[3, 0]

            # Predict line of sight and deploy in measurement matrix, (9.144)
            H_matrix[[j], 0:3] = -np.transpose(delta_r) / range_
            H_matrix[j, 3] = 1

        # Unweighted least-squares solution, (9.35)/(9.141)
        meas_innovation = GNSS_measurements[:, [0]] - pred_meas
        x_est = x_pred + np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(H_matrix), H_matrix)), np.transpose(H_matrix)), meas_innovation)

        # Test convergence
        test_convergence = math.sqrt(np.matmul(np.transpose(x_est - x_pred), (x_est - x_pred)))

        # Set predictions to estimates for next iteration
        x_pred = x_est

    # Set outputs to estimates
    est_r_ea_e = x_est[0:3, [0]]
    est_clock = np.zeros((2, 1))
    est_clock[0, 0] = x_est[3, 0]

    # VELOCITY AND CLOCK DRIFT

    # Skew symmetric matrix of Earth rate
    Omega_ie = skew_symmetric(np.array([[0], [0], [omega_ie]]))

    # Setup predicted state
    x_pred = np.zeros((4, 1))
    x_pred[0:3, [0]] = predicted_v_ea_e
    test_convergence = 1

    # Repeat until convergence
    while test_convergence > 0.0001:
        # Loop measurements
        pred_meas = np.zeros((no_GNSS_meas, 1))
        for j in range(0, no_GNSS_meas):
            # Predict approximate range
            delta_r = np.transpose(GNSS_measurements[[j], 2:5]) - est_r_ea_e
            approx_range = math.sqrt(np.matmul(np.transpose(delta_r), delta_r))

            # Calculate frame rotation during signal transit time using (8.36)
            C_e_I = np.zeros((3, 3))
            C_e_I[0, 0] = 1
            C_e_I[0, 1] = omega_ie * approx_range / c
            C_e_I[1, 0] = -omega_ie * approx_range / c
            C_e_I[1, 1] = 1
            C_e_I[2, 2] = 1

            # Calculate range using (8.35)
            delta_r = np.matmul(C_e_I, np.transpose(GNSS_measurements[[j], 2:5])) - est_r_ea_e
            range_ = math.sqrt(np.matmul(np.transpose(delta_r), delta_r))

            # Calculate line of sight using (8.41)
            u_as_e = delta_r / range_

            # Predict pseudo-range rate using (9.143)
            temp = np.matmul(C_e_I, (np.transpose(GNSS_measurements[[j], 5:]) + np.matmul(Omega_ie, np.transpose(GNSS_measurements[[j], 2:5])))) -\
                   (x_pred[0:3, [0]] + np.matmul(Omega_ie, est_r_ea_e))
            range_rate = np.matmul(np.transpose(u_as_e), temp)
            pred_meas[j, 0] = range_rate + x_pred[3, 0]

            # Predict line of sight and deploy in measurement matrix, (9.144)
            H_matrix[[j], 0:3] = -np.transpose(u_as_e)
            H_matrix[j, 3] = 1

        # Unweighted least-squares solution, (9.35)/(9.141)
        meas_innovation = GNSS_measurements[:, [1]] - pred_meas
        x_est = x_pred + np.matmul(
                np.matmul(np.linalg.inv(np.matmul(np.transpose(H_matrix), H_matrix)), np.transpose(H_matrix)),
                meas_innovation)

        # Test convergence
        test_convergence = math.sqrt(np.matmul(np.transpose(x_est - x_pred), (x_est - x_pred)))

        # Set predictions to estimates for next iteration
        x_pred = x_est

    # Set outputs to estimates
    est_v_ea_e = x_est[0:3, [0]]
    est_clock[1, 0] = x_est[3, 0]

    return est_r_ea_e, est_v_ea_e, est_clock

