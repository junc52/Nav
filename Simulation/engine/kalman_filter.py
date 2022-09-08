# This software is distributed under a Modified BSD License as follows:
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
from Simulation.utils.frame_transform import skew_symmetric
from Simulation.utils.gravitation_gravity_model import gravity_ecef


def gnss_kf_epoch(GNSS_measurements, no_meas, tor_s, x_est_old, P_matrix_old, GNSS_KF_config):
    """
    GNSS_KF_Epoch - Implements one cycle of the GNSS extended Kalman filter

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 12/4/2012 by Paul Groves

    Inputs:
      GNSS_measurements     GNSS measurement data:
        Column 1              Pseudo-range measurements (m)
        Column 2              Pseudo-range rate measurements (m/s)
        Columns 3-5           Satellite ECEF position (m)
        Columns 6-8           Satellite ECEF velocity (m/s)
      no_meas               Number of satellites for which measurements are
                            supplied
      tor_s                 propagation interval (s)
      x_est_old             previous Kalman filter state estimates
      P_matrix_old          previous Kalman filter error covariance matrix
      GNSS_KF_config
        .accel_PSD              Acceleration PSD per axis (m^2/s^3)
        .clock_freq_PSD         Receiver clock frequency-drift PSD (m^2/s^3)
        .clock_phase_PSD        Receiver clock phase-drift PSD (m^2/s)
        .pseudo_range_SD        Pseudo-range measurement noise SD (m)
        .range_rate_SD          Pseudo-range rate measurement noise SD (m/s)

    Outputs:
      x_est_new             updated Kalman filter state estimates
        Columns 1-3            estimated ECEF user position (m)
        Columns 4-6            estimated ECEF user velocity (m/s)
        Column 7               estimated receiver clock offset (m)
        Column 8               estimated receiver clock drift (m/s)
      P_matrix_new          updated Kalman filter error covariance matrix


    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """

    # CONSTANTS
    c = 299792458           # Speed of light in m / s
    omega_ie = 7.292115E-5  # Earth rotation rate in rad / s

    # SYSTEM PROPAGATION PHASE
    # 1. Determine transition matrix using (9.147) and (9.150)
    Phi_matrix = np.identity(8)
    Phi_matrix[0, 3] = tor_s
    Phi_matrix[1, 4] = tor_s
    Phi_matrix[2, 5] = tor_s
    Phi_matrix[6, 7] = tor_s

    # 2. Determine system noise covariance matrix using (9.152)
    Q_matrix = np.zeros((8, 8))
    Q_matrix[0: 3, 0: 3] = np.identity(3) * (GNSS_KF_config.accel_PSD * tor_s ** 3 / 3)
    Q_matrix[0: 3, 3: 6] = np.identity(3) * (GNSS_KF_config.accel_PSD * tor_s ** 2 / 2)
    Q_matrix[3: 6, 0: 3] = np.identity(3) * (GNSS_KF_config.accel_PSD * tor_s ** 2 / 2)
    Q_matrix[3: 6, 3: 6] = np.identity(3) * (GNSS_KF_config.accel_PSD * tor_s)
    Q_matrix[6, 6] = (GNSS_KF_config.clock_freq_PSD * tor_s ** 3 / 3) + GNSS_KF_config.clock_phase_PSD * tor_s
    Q_matrix[6, 7] = GNSS_KF_config.clock_freq_PSD * tor_s ** 2 / 2
    Q_matrix[7, 6] = GNSS_KF_config.clock_freq_PSD * tor_s ** 2 / 2
    Q_matrix[7, 7] = GNSS_KF_config.clock_freq_PSD * tor_s

    # 3. Propagate state estimates using (3.14)
    x_est_propagated = np.matmul(Phi_matrix, x_est_old)

    # 4. Propagate state estimation error covariance matrix using (3.15)
    P_matrix_propagated = np.matmul(np.matmul(Phi_matrix,P_matrix_old), np.transpose(Phi_matrix)) + Q_matrix

    # MEASUREMENTS UPDATE PHASE
    # Skew symmetric matrix of Earth rate
    Omega_ie = skew_symmetric(np.array([[0], [0], [omega_ie]]))

    u_as_e_T = np.zeros((no_meas, 3))
    pred_meas = np.zeros((no_meas, 2))

    # Loop measurements
    for j in range(0, no_meas):
        # Predict approximate range
        delta_r = np.transpose(GNSS_measurements[j, 2:5]) - x_est_propagated[0:3, 0]
        approx_range = math.sqrt(np.matmul(np.transpose(delta_r), delta_r))

        # Calculate frame rotation during signal transit time using (8.36)
        C_e_I = np.zeros((3, 3))
        C_e_I[0, 0] = 1
        C_e_I[0, 1] = omega_ie * approx_range / c
        C_e_I[1, 0] = -omega_ie * approx_range / c
        C_e_I[1, 1] = 1
        C_e_I[2, 2] = 1

        # Predict pseudo-range using (9.165)
        delta_r = np.matmul(C_e_I, np.transpose(GNSS_measurements[j, 2:5])) - x_est_propagated[0:3, 0]
        range_ = math.sqrt(np.matmul(np.transpose(delta_r), delta_r))
        pred_meas[j, 0] = range_ + x_est_propagated[6, 0]

        # Predict line of sight using (8.41)
        u_as_e_T[[j], 0:3] = np.transpose(delta_r) / range_

        # Predict pseudo-range rate using (9.165)
        temp = np.matmul(C_e_I, (np.transpose(GNSS_measurements[[j], 5:]) + np.matmul(Omega_ie, np.transpose(
            GNSS_measurements[[j], 2:5])))) - (x_est_propagated[3:6, [0]] + np.matmul(Omega_ie, x_est_propagated[0:3, [0]]))
        range_rate = np.matmul(u_as_e_T[[j], 0:3], temp)
        pred_meas[j, 1] = range_rate + x_est_propagated[7, 0]

    # 5. Set-up measurement matrix using (9.163)
    H_matrix = np.zeros((no_meas * 2, 8))
    H_matrix[0: no_meas, 0: 3] = -u_as_e_T[0: no_meas, 0: 3]
    H_matrix[0: no_meas, [6]] = np.ones((no_meas, 1))
    H_matrix[no_meas: (2 * no_meas), 3: 6] = -u_as_e_T[0: no_meas, 0: 3]
    H_matrix[no_meas: (2 * no_meas), [7]] = np.ones((no_meas, 1))

    # 6. Set-up measurement noise covariance matrix assuming all measurements
    #    are independent and have equal variance for a given measurement type
    R_matrix = np.zeros((no_meas * 2, no_meas * 2))
    R_matrix[0: no_meas, 0: no_meas] = np.identity(no_meas) * GNSS_KF_config.pseudo_range_SD ** 2
    R_matrix[no_meas: (2 * no_meas), no_meas: (2 * no_meas)] = np.identity(no_meas) * GNSS_KF_config.range_rate_SD ** 2

    # 7. Calculate Kalman gain using (3.21)
    PH_T = np.matmul(P_matrix_propagated, np.transpose(H_matrix))
    HPH_T = np.matmul(np.matmul(H_matrix, P_matrix_propagated), np.transpose(H_matrix))
    K_matrix = np.matmul(PH_T, np.linalg.inv(HPH_T + R_matrix))

    # 8. Formulate measurement innovations using (3.88)
    delta_z = np.zeros((no_meas * 2, 1))
    delta_z[0: no_meas, [0]] = GNSS_measurements[0: no_meas, [0]] - pred_meas[0: no_meas, [0]]
    delta_z[no_meas: (2 * no_meas), [0]] = GNSS_measurements[0: no_meas, [1]] - pred_meas[0: no_meas, [1]]

    # 9. Update state estimates using (3.24)
    x_est_new = x_est_propagated + np.matmul(K_matrix, delta_z)

    # 10. Update state estimation error covariance matrix using (3.25)
    P_matrix_new = np.matmul(np.identity(8) - np.matmul(K_matrix, H_matrix), P_matrix_propagated)

    return x_est_new, P_matrix_new


def tc_kf_epoch(GNSS_measurements, no_meas, tor_s, est_C_b_e_old, est_v_eb_e_old, est_r_eb_e_old,
                est_IMU_bias_old, est_clock_old, P_matrix_old, meas_f_ib_b, est_L_b_old, TC_KF_config):
    """
    TC_KF_Epoch - Implements one cycle of the tightly coupled INS/GNSS
    extended Kalman filter plus closed-loop correction of all inertial states

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 12/4/2012 by Paul Groves

    Inputs:
      GNSS_measurements     GNSS measurement data:
        Column 1              Pseudo-range measurements (m)
        Column 2              Pseudo-range rate measurements (m/s)
        Columns 3-5           Satellite ECEF position (m)
        Columns 6-8           Satellite ECEF velocity (m/s)
      no_meas               Number of satellites for which measurements are
                            supplied
      tor_s                 propagation interval (s)
      est_C_b_e_old         prior estimated body to ECEF coordinate
                            transformation matrix
      est_v_eb_e_old        prior estimated ECEF user velocity (m/s)
      est_r_eb_e_old        prior estimated ECEF user position (m)
      est_IMU_bias_old      prior estimated IMU biases (body axes)
      est_clock_old         prior Kalman filter state estimates
      P_matrix_old          previous Kalman filter error covariance matrix
      meas_f_ib_b           measured specific force
      est_L_b_old           previous latitude solution
      TC_KF_config
        .gyro_noise_PSD     Gyro noise PSD (rad^2/s)
        .accel_noise_PSD    Accelerometer noise PSD (m^2 s^-3)
        .accel_bias_PSD     Accelerometer bias random walk PSD (m^2 s^-5)
        .gyro_bias_PSD      Gyro bias random walk PSD (rad^2 s^-3)
        .clock_freq_PSD     Receiver clock frequency-drift PSD (m^2/s^3)
        .clock_phase_PSD    Receiver clock phase-drift PSD (m^2/s)
        .pseudo_range_SD    Pseudo-range measurement noise SD (m)
        .range_rate_SD      Pseudo-range rate measurement noise SD (m/s)

    Outputs:
      est_C_b_e_new     updated estimated body to ECEF coordinate
                      transformation matrix
      est_v_eb_e_new    updated estimated ECEF user velocity (m/s)
      est_r_eb_e_new    updated estimated ECEF user position (m)
      est_IMU_bias_new  updated estimated IMU biases
        Rows 1-3          estimated accelerometer biases (m/s^2)
        Rows 4-6          estimated gyro biases (rad/s)
      est_clock_new     updated Kalman filter state estimates
        Row 1             estimated receiver clock offset (m)
        Row 2             estimated receiver clock drift (m/s)
      P_matrix_new      updated Kalman filter error covariance matrix


    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    # CONSTANTS
    c = 299792458           # Speed of light in m / s
    omega_ie = 7.292115E-5  # Earth rotation rate in rad / s
    R_0 = 6378137           # WGS84 Equatorial radius in meters
    e = 0.0818191908425     # WGS84 eccentricity

    # Skew symmetric matrix of Earth rate
    Omega_ie = skew_symmetric(np.array([[0], [0], [omega_ie]]))

    # SYSTEM PROPAGATION PHASE
    # 1. Determine transition matrix using (14.50)
    Phi_matrix = np.identity(17)
    Phi_matrix[0:3, 0:3] = Phi_matrix[0:3, 0:3] - Omega_ie * tor_s
    Phi_matrix[0:3, 12:15] = est_C_b_e_old * tor_s
    Phi_matrix[3:6, 0:3] = -tor_s * skew_symmetric(np.matmul(est_C_b_e_old, meas_f_ib_b))
    Phi_matrix[3:6, 3:6] = Phi_matrix[3:6, 3:6] - 2 * Omega_ie * tor_s

    # Radius of curvature for east-west motion from (2.106)
    R_E_L = R_0 / math.sqrt(1 - (e * math.sin(est_L_b_old)) ** 2)

    # Geocentric radius from (2.137)
    geocentric_radius = R_E_L * math.sqrt(math.cos(est_L_b_old) ** 2 + (1 - e ** 2) ** 2 * math.sin(est_L_b_old) ** 2)

    Phi_matrix[3: 6, 6: 9] = -tor_s * np.matmul(((2 / geocentric_radius) * gravity_ecef(est_r_eb_e_old)), (np.transpose(est_r_eb_e_old) / math.sqrt (np.matmul(np.transpose(est_r_eb_e_old), est_r_eb_e_old))))
    Phi_matrix[3: 6, 9: 12] = est_C_b_e_old * tor_s
    Phi_matrix[6: 9, 3: 6] = np.identity(3) * tor_s
    Phi_matrix[15, 16] = tor_s

    # 2. Determine approximate system noise covariance matrix using (14.82)
    Q_prime_matrix = np.zeros((17, 17))
    Q_prime_matrix[0: 3, 0: 3] = np.identity(3) * TC_KF_config.gyro_noise_PSD * tor_s
    Q_prime_matrix[3: 6, 3: 6] = np.identity(3) * TC_KF_config.accel_noise_PSD * tor_s
    Q_prime_matrix[9: 12, 9: 12] = np.identity(3) * TC_KF_config.accel_bias_PSD * tor_s
    Q_prime_matrix[12: 15, 12: 15] = np.identity(3) * TC_KF_config.gyro_bias_PSD * tor_s
    Q_prime_matrix[15, 15] = TC_KF_config.clock_phase_PSD * tor_s
    Q_prime_matrix[16, 16] = TC_KF_config.clock_freq_PSD * tor_s

    # 3. Propagate state estimates using (3.14) noting that only the clock
    # states are non-zero due to closed-loop correction
    x_est_propagated = np.zeros((17, 1))
    x_est_propagated[15, 0] = est_clock_old[0, 0] + est_clock_old[1, 0] * tor_s
    x_est_propagated[16, 0] = est_clock_old[1, 0]

    # 4. Propagate state estimation error covariance matrix using (3.46)
    P_matrix_propagated = np.matmul(np.matmul(Phi_matrix, (P_matrix_old + 0.5 * Q_prime_matrix)), np.transpose(Phi_matrix)) + 0.5 * Q_prime_matrix

    # MEASUREMENTS UPDATE PHASE

    u_as_e_T = np.zeros((no_meas, 3))
    pred_meas = np.zeros((no_meas, 2))

    # Loop measurements
    for j in range(0, no_meas):
        # Predict approximate range
        delta_r = np.transpose(GNSS_measurements[[j], 2:5]) - est_r_eb_e_old
        approx_range = math.sqrt(np.matmul(np.transpose(delta_r), delta_r))

        # Calculate frame rotation during signal transit time using (8.36)
        C_e_I = np.zeros((3, 3))
        C_e_I[0, 0] = 1
        C_e_I[0, 1] = omega_ie * approx_range / c
        C_e_I[1, 0] = -omega_ie * approx_range / c
        C_e_I[1, 1] = 1
        C_e_I[2, 2] = 1

        # Predict pseudo-range using (9.165)
        delta_r = np.matmul(C_e_I, np.transpose(GNSS_measurements[[j], 2:5])) - est_r_eb_e_old
        range_ = math.sqrt(np.matmul(np.transpose(delta_r), delta_r))
        pred_meas[j, 0] = range_ + x_est_propagated[15, 0]

        # Predict line of sight using (8.41)
        u_as_e_T[[j], 0:3] = np.transpose(delta_r) / range_

        # Predict pseudo-range rate using (9.165)
        temp = np.matmul(C_e_I, (np.transpose(GNSS_measurements[[j], 5:]) + np.matmul(Omega_ie, np.transpose(
            GNSS_measurements[[j], 2:5])))) - (est_v_eb_e_old + np.matmul(Omega_ie, est_r_eb_e_old))
        range_rate = np.matmul(u_as_e_T[[j], 0:3], temp)
        pred_meas[j, 1] = range_rate + x_est_propagated[16, 0]

    # 5. Set-up measurement matrix using (14.126)
    H_matrix = np.zeros((no_meas * 2, 17))
    H_matrix[0: no_meas, 6: 9] = u_as_e_T[0: no_meas, 0: 3]
    H_matrix[0: no_meas, [15]] = np.ones((no_meas, 1))
    H_matrix[no_meas: (2 * no_meas), 3: 6] = u_as_e_T[0: no_meas, 0: 3]
    H_matrix[no_meas: (2 * no_meas), [16]] = np.ones((no_meas, 1))

    # 6. Set-up measurement noise covariance matrix assuming all measurements
    #    are independent and have equal variance for a given measurement type
    R_matrix = np.zeros((no_meas * 2, no_meas * 2))
    R_matrix[0: no_meas, 0: no_meas] = np.identity(no_meas) * TC_KF_config.pseudo_range_SD ** 2
    R_matrix[no_meas: (2 * no_meas), no_meas: (2 * no_meas)] = np.identity(no_meas) * TC_KF_config.range_rate_SD ** 2

    # 7. Calculate Kalman gain using (3.21)
    PH_T = np.matmul(P_matrix_propagated, np.transpose(H_matrix))
    HPH_T = np.matmul(np.matmul(H_matrix, P_matrix_propagated), np.transpose(H_matrix))
    K_matrix = np.matmul(PH_T, np.linalg.inv(HPH_T + R_matrix))

    # 8. Formulate measurement innovations using (14.119)
    delta_z = np.zeros((no_meas * 2, 1))
    delta_z[0: no_meas, [0]] = GNSS_measurements[0: no_meas, [0]] - pred_meas[0: no_meas, [0]]
    delta_z[no_meas: (2 * no_meas), [0]] = GNSS_measurements[0: no_meas, [1]] - pred_meas[0: no_meas, [1]]

    # 9. Update state estimates using (3.24)
    x_est_new = x_est_propagated + np.matmul(K_matrix, delta_z)

    # 10. Update state estimation error covariance matrix using (3.25)
    P_matrix_new = np.matmul(np.identity(17) - np.matmul(K_matrix, H_matrix), P_matrix_propagated)

    # CLOSED-LOOP CORRECTION

    # Correct attitude, velocity, and position using (14.7-9)
    est_C_b_e_new = np.matmul((np.identity(3) - skew_symmetric(x_est_new[0:3, [0]])), est_C_b_e_old)
    est_v_eb_e_new = est_v_eb_e_old - x_est_new[3:6, [0]]
    est_r_eb_e_new = est_r_eb_e_old - x_est_new[6:9, [0]]

    # Update IMU bias and GNSS receiver clock estimates
    est_IMU_bias_new = est_IMU_bias_old + x_est_new[9:15, [0]]
    est_clock_new = x_est_new[15:17, [0]]

    return est_C_b_e_new, est_v_eb_e_new, est_r_eb_e_new, est_IMU_bias_new, est_clock_new, P_matrix_new


def lc_kf_epoch(GNSS_r_eb_e, GNSS_v_eb_e, tor_s, est_C_b_e_old, est_v_eb_e_old, est_r_eb_e_old, est_IMU_bias_old, P_matrix_old, meas_f_ib_b, est_L_b_old, LC_KF_config):
    """
    LC_KF_Epoch - Implements one cycle of the loosely coupled INS/GNSS
    Kalman filter plus closed-loop correction of all inertial states

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 12/4/2012 by Paul Groves

    Inputs:
      GNSS_r_eb_e           GNSS estimated ECEF user position (m)
      GNSS_v_eb_e           GNSS estimated ECEF user velocity (m/s)
      tor_s                 propagation interval (s)
      est_C_b_e_old         prior estimated body to ECEF coordinate
                            transformation matrix
      est_v_eb_e_old        prior estimated ECEF user velocity (m/s)
      est_r_eb_e_old        prior estimated ECEF user position (m)
      est_IMU_bias_old      prior estimated IMU biases (body axes)
      P_matrix_old          previous Kalman filter error covariance matrix
      meas_f_ib_b           measured specific force
      est_L_b_old           previous latitude solution
      LC_KF_config
        .gyro_noise_PSD     Gyro noise PSD (rad^2/s)
        .accel_noise_PSD    Accelerometer noise PSD (m^2 s^-3)
        .accel_bias_PSD     Accelerometer bias random walk PSD (m^2 s^-5)
        .gyro_bias_PSD      Gyro bias random walk PSD (rad^2 s^-3)
        .pos_meas_SD            Position measurement noise SD per axis (m)
        .vel_meas_SD            Velocity measurement noise SD per axis (m/s)

    Outputs:
      est_C_b_e_new     updated estimated body to ECEF coordinate
                      transformation matrix
      est_v_eb_e_new    updated estimated ECEF user velocity (m/s)
      est_r_eb_e_new    updated estimated ECEF user position (m)
      est_IMU_bias_new  updated estimated IMU biases
        Rows 1-3          estimated accelerometer biases (m/s^2)
        Rows 4-6          estimated gyro biases (rad/s)
      P_matrix_new      updated Kalman filter error covariance matrix


    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    # CONSTANTS
    c = 299792458  # Speed of light in m / s
    omega_ie = 7.292115E-5  # Earth rotation rate in rad / s
    R_0 = 6378137  # WGS84 Equatorial radius in meters
    e = 0.0818191908425  # WGS84 eccentricity

    # Skew symmetric matrix of Earth rate
    Omega_ie = skew_symmetric(np.array([[0], [0], [omega_ie]]))

    # SYSTEM PROPAGATION PHASE
    # 1. Determine transition matrix using (14.50) (first-order approx)
    Phi_matrix = np.identity(15)
    Phi_matrix[0:3, 0:3] = Phi_matrix[0:3, 0:3] - Omega_ie * tor_s
    Phi_matrix[0:3, 12:15] = est_C_b_e_old * tor_s
    Phi_matrix[3:6, 0:3] = -tor_s * skew_symmetric(np.matmul(est_C_b_e_old, meas_f_ib_b))
    Phi_matrix[3:6, 3:6] = Phi_matrix[3:6, 3:6] - 2 * Omega_ie * tor_s

    # Radius of curvature for east-west motion from (2.106)
    R_E_L = R_0 / math.sqrt(1 - (e * math.sin(est_L_b_old)) ** 2)

    # Geocentric radius from (2.137)
    geocentric_radius = R_E_L * math.sqrt(math.cos(est_L_b_old) ** 2 + (1 - e ** 2) ** 2 * math.sin(est_L_b_old) ** 2)

    Phi_matrix[3: 6, 6: 9] = -tor_s * np.matmul(((2 / geocentric_radius) * gravity_ecef(est_r_eb_e_old)), (np.transpose(
        est_r_eb_e_old) / math.sqrt(np.matmul(np.transpose(est_r_eb_e_old), est_r_eb_e_old))))
    Phi_matrix[3: 6, 9: 12] = est_C_b_e_old * tor_s
    Phi_matrix[6: 9, 3: 6] = np.identity(3) * tor_s

    # 2. Determine approximate system noise covariance matrix using (14.82)
    Q_prime_matrix = np.zeros((15, 15))
    Q_prime_matrix[0: 3, 0: 3] = np.identity(3) * LC_KF_config.gyro_noise_PSD * tor_s
    Q_prime_matrix[3: 6, 3: 6] = np.identity(3) * LC_KF_config.accel_noise_PSD * tor_s
    Q_prime_matrix[9: 12, 9: 12] = np.identity(3) * LC_KF_config.accel_bias_PSD * tor_s
    Q_prime_matrix[12: 15, 12: 15] = np.identity(3) * LC_KF_config.gyro_bias_PSD * tor_s

    # 3. Propagate state estimates using (3.14) noting that only the clock
    # states are non-zero due to closed-loop correction
    x_est_propagated = np.zeros((15, 1))

    # 4. Propagate state estimation error covariance matrix using (3.46)
    P_matrix_propagated = np.matmul(np.matmul(Phi_matrix, (P_matrix_old + 0.5 * Q_prime_matrix)),
                                    np.transpose(Phi_matrix)) + 0.5 * Q_prime_matrix

    # MEASUREMENTS UPDATE PHASE

    # 5. Set-up measurement matrix using (14.115)
    H_matrix = np.zeros((6, 15))
    H_matrix[0: 3, 6: 9] = -np.identity(3)
    H_matrix[3: 6, 3: 6] = -np.identity(3)

    # 6. Set-up measurement noise covariance matrix assuming all measurements
    #    are independent and have equal variance for a given measurement type
    R_matrix = np.zeros((6, 6))
    R_matrix[0: 3, 0: 3] = np.identity(3) * LC_KF_config.pos_meas_SD ** 2
    R_matrix[3: 6, 3: 6] = np.identity(3) * LC_KF_config.vel_meas_SD ** 2

    # 7. Calculate Kalman gain using (3.21)
    PH_T = np.matmul(P_matrix_propagated, np.transpose(H_matrix))
    HPH_T = np.matmul(np.matmul(H_matrix, P_matrix_propagated), np.transpose(H_matrix))
    K_matrix = np.matmul(PH_T, np.linalg.inv(HPH_T + R_matrix))

    # 8. Formulate measurement innovations using (14.102), noting that zero
    #    lever arm is assumed here
    delta_z = np.zeros((6, 1))
    delta_z[0: 3, [0]] = GNSS_r_eb_e -est_r_eb_e_old
    delta_z[3:, [0]] = GNSS_v_eb_e - est_v_eb_e_old

    # 9. Update state estimates using (3.24)
    x_est_new = x_est_propagated + np.matmul(K_matrix, delta_z)

    # 10. Update state estimation error covariance matrix using (3.25)
    P_matrix_new = np.matmul(np.identity(15) - np.matmul(K_matrix, H_matrix), P_matrix_propagated)

    # CLOSED-LOOP CORRECTION

    # Correct attitude, velocity, and position using (14.7-9)
    est_C_b_e_new = np.matmul((np.identity(3) - skew_symmetric(x_est_new[0:3, [0]])), est_C_b_e_old)
    est_v_eb_e_new = est_v_eb_e_old - x_est_new[3:6, [0]]
    est_r_eb_e_new = est_r_eb_e_old - x_est_new[6:9, [0]]

    # Update IMU bias and GNSS receiver clock estimates
    est_IMU_bias_new = est_IMU_bias_old + x_est_new[9:15, [0]]

    return est_C_b_e_new, est_v_eb_e_new, est_r_eb_e_new, est_IMU_bias_new, P_matrix_new
