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
from Simulation.utils.frame_transform import *
from Simulation.engine.calculate_errors import calculate_errors_ned
from Simulation.engine.gnss_ls_position_velocity import gnss_ls_position_velocity
from Simulation.engine.initialize import initialize_tc_P_matrix, initialize_lc_P_matrix
from Simulation.engine.initialize_models import *
from Simulation.engine.kalman_filter import tc_kf_epoch, lc_kf_epoch
from Simulation.engine.kinematics import *
from Simulation.engine.nav_equations import *


def tightly_coupled_ins_gnss(in_profile, no_epochs, initialization_errors, IMU_errors, GNSS_config, TC_KF_config):
    """
    Tightly_coupled_INS_GNSS - Simulates inertial navigation using ECEF
    navigation equations and kinematic model, GNSS and tightly coupled
    INS/GNSS integration.

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 12/4/2012 by Paul Groves

    Inputs:
      in_profile   True motion profile array
      no_epochs    Number of epochs of profile data
      initialization_errors
        .delta_r_eb_n     position error resolved along NED (m)
        .delta_v_eb_n     velocity error resolved along NED (m/s)
        .delta_eul_nb_n   attitude error as NED Euler angles (rad)
      IMU_errors
        .delta_r_eb_n     position error resolved along NED (m)
        .b_a              Accelerometer biases (m/s^2)
        .b_g              Gyro biases (rad/s)
        .M_a              Accelerometer scale factor and cross coupling errors
        .M_g              Gyro scale factor and cross coupling errors
        .G_g              Gyro g-dependent biases (rad-sec/m)
        .accel_noise_root_PSD   Accelerometer noise root PSD (m s^-1.5)
        .gyro_noise_root_PSD    Gyro noise root PSD (rad s^-0.5)
        .accel_quant_level      Accelerometer quantization level (m/s^2)
        .gyro_quant_level       Gyro quantization level (rad/s)
      GNSS_config
        .epoch_interval     Interval between GNSS epochs (s)
        .init_est_r_ea_e    Initial estimated position (m; ECEF)
        .no_sat             Number of satellites in constellation
        .r_os               Orbital radius of satellites (m)
        .inclination        Inclination angle of satellites (deg)
        .const_delta_lambda Longitude offset of constellation (deg)
        .const_delta_t      Timing offset of constellation (s)
        .mask_angle         Mask angle (deg)
        .SIS_err_SD         Signal in space error SD (m)
        .zenith_iono_err_SD Zenith ionosphere error SD (m)
        .zenith_trop_err_SD Zenith troposphere error SD (m)
        .code_track_err_SD  Code tracking error SD (m)
        .rate_track_err_SD  Range rate tracking error SD (m/s)
        .rx_clock_offset    Receiver clock offset at time=0 (m)
        .rx_clock_drift     Receiver clock drift at time=0 (m/s)
      TC_KF_config
        .init_att_unc           Initial attitude uncertainty per axis (rad)
        .init_vel_unc           Initial velocity uncertainty per axis (m/s)
        .init_pos_unc           Initial position uncertainty per axis (m)
        .init_b_a_unc           Initial accel. bias uncertainty (m/s^2)
        .init_b_g_unc           Initial gyro. bias uncertainty (rad/s)
        .init_clock_offset_unc  Initial clock offset uncertainty per axis (m)
        .init_clock_drift_unc   Initial clock drift uncertainty per axis (m/s)
        .gyro_noise_PSD         Gyro noise PSD (rad^2/s)
        .accel_noise_PSD        Accelerometer noise PSD (m^2 s^-3)
        .accel_bias_PSD         Accelerometer bias random walk PSD (m^2 s^-5)
        .gyro_bias_PSD          Gyro bias random walk PSD (rad^2 s^-3)
        .clock_freq_PSD         Receiver clock frequency-drift PSD (m^2/s^3)
        .clock_phase_PSD        Receiver clock phase-drift PSD (m^2/s)
        .pseudo_range_SD        Pseudo-range measurement noise SD (m)
        .range_rate_SD          Pseudo-range rate measurement noise SD (m/s)

    Outputs:
      out_profile        Navigation solution as a motion profile array
      out_errors         Navigation solution error array
      out_IMU_bias_est   Kalman filter IMU bias estimate array
      out_clock          GNSS Receiver clock estimate array
      out_KF_SD          Output Kalman filter state uncertainties

    Format of motion profiles:
     Column 1: time (sec)
     Column 2: latitude (rad)
     Column 3: longitude (rad)
     Column 4: height (m)
     Column 5: north velocity (m/s)
     Column 6: east velocity (m/s)
     Column 7: down velocity (m/s)
     Column 8: roll angle of body w.r.t NED (rad)
     Column 9: pitch angle of body w.r.t NED (rad)
     Column 10: yaw angle of body w.r.t NED (rad)

    Format of error array:
     Column 1: time (sec)
     Column 2: north position error (m)
     Column 3: east position error (m)
     Column 4: down position error (m)
     Column 5: north velocity error (m/s)
     Column 6: east velocity error (m/s)
     Column 7: down velocity (error m/s)
     Column 8: attitude error about north (rad)
     Column 9: attitude error about east (rad)
     Column 10: attitude error about down = heading error  (rad)

    Format of output IMU biases array:
     Column 1: time (sec)
     Column 2: estimated X accelerometer bias (m/s^2)
     Column 3: estimated Y accelerometer bias (m/s^2)
     Column 4: estimated Z accelerometer bias (m/s^2)
     Column 5: estimated X gyro bias (rad/s)
     Column 6: estimated Y gyro bias (rad/s)
     Column 7: estimated Z gyro bias (rad/s)

    Format of receiver clock array:
     Column 1: time (sec)
     Column 2: estimated clock offset (m)
     Column 3: estimated clock drift (m/s)

    Format of KF state uncertainties array:
     Column 1: time (sec)
     Column 2: X attitude error uncertainty (rad)
     Column 3: Y attitude error uncertainty (rad)
     Column 4: Z attitude error uncertainty (rad)
     Column 5: X velocity error uncertainty (m/s)
     Column 6: Y velocity error uncertainty (m/s)
     Column 7: Z velocity error uncertainty (m/s)
     Column 8: X position error uncertainty (m)
     Column 9: Y position error uncertainty (m)
     Column 10: Z position error uncertainty (m)
     Column 11: X accelerometer bias uncertainty (m/s^2)
     Column 12: Y accelerometer bias uncertainty (m/s^2)
     Column 13: Z accelerometer bias uncertainty (m/s^2)
     Column 14: X gyro bias uncertainty (rad/s)
     Column 15: Y gyro bias uncertainty (rad/s)
     Column 16: Z gyro bias uncertainty (rad/s)
     Column 17: clock offset uncertainty (m)
     Column 18: clock drift uncertainty (m/s)

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    # Initialize true navigation solution
    old_time = in_profile[0, 0]                                   # time (sec)
    true_L_b = in_profile[0, 1]                             # latitude (rad)
    true_lambda_b = in_profile[0, 2]                        # longitude (rad)
    true_h_b = in_profile[0, 3]
    true_v_eb_n = np.transpose(in_profile[[0], 4:7])        # velocity of body frame wrt ECEF resolved in NED frame (m/s)
    true_eul_nb = np.transpose(in_profile[[0], 7:])           # euler angle of body frame wrt NED frame (rad)
    true_C_b_n = np.transpose(euler_to_ctm(true_eul_nb))    # rotation matrix from body to NED frame

    old_true_r_eb_e, old_true_v_eb_e, old_true_C_b_e = ned_to_ecef(true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)

    # Determine satellite positions and velocities
    sat_r_es_e, sat_v_es_e = satellite_positions_and_velocities(old_time, GNSS_config)

    # Initialize the GNSS biases. Note that these are assumed constant throughout
    # the simulation and are based on the initial elevation angles.Therefore,
    # this function is unsuited to simulations longer than about 30 min.
    GNSS_biases = initialize_gnss_biases(sat_r_es_e, old_true_r_eb_e, true_L_b, true_lambda_b, GNSS_config)

    # Generate GNSS measurements
    GNSS_measurements, no_GNSS_meas = generate_gnss_measurements(old_time, sat_r_es_e, sat_v_es_e, old_true_r_eb_e,
                                                                 true_L_b, true_lambda_b, old_true_v_eb_e,
                                                                 GNSS_biases, GNSS_config)

    # Determine Least-squares GNSS position solution
    old_est_r_eb_e, old_est_v_eb_e, est_clock = gnss_ls_position_velocity(GNSS_measurements, no_GNSS_meas,
                                                                  GNSS_config.init_est_r_ea_e, np.zeros((3, 1)))

    old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n = pv_ecef_to_ned(old_est_r_eb_e, old_est_v_eb_e)
    est_L_b = old_est_L_b

    # Initialize estimated attitude solution
    old_est_C_b_n = initialize_ned_attitude(true_C_b_n, initialization_errors)
    temp1, temp2, old_est_C_b_e = ned_to_ecef(old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_n)

    # Initialize output profile record and errors record
    out_profile = np.zeros((no_epochs, 10))
    out_errors = np.zeros((no_epochs, 10))
    out_clock = np.zeros((no_epochs, 3))
    out_KF_SD = np.zeros((no_epochs, 18))
    out_IMU_bias_est = np.zeros((no_epochs, 7))

    # Generate output profile record
    out_profile[0, 0] = old_time
    out_profile[0, 1] = old_est_L_b
    out_profile[0, 2] = old_est_lambda_b
    out_profile[0, 3] = old_est_h_b
    out_profile[[0], 4: 7] = np.transpose(old_est_v_eb_n)
    out_profile[[0], 7:] = np.transpose(ctm_to_euler(np.transpose(old_est_C_b_n)))

    # Determine errors and generate output record
    delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = calculate_errors_ned(
        old_est_L_b,old_est_lambda_b,old_est_h_b,old_est_v_eb_n,old_est_C_b_n, true_L_b,
        true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)
    out_errors[0, 0] = old_time
    out_errors[[0], 1: 4] = np.transpose(delta_r_eb_n)
    out_errors[[0], 4: 7] = np.transpose(delta_v_eb_n)
    out_errors[[0], 7:] = np.transpose(delta_eul_nb_n)

    # Initialize Kalman filter P matrix and IMU bias states
    P_matrix = initialize_tc_P_matrix(TC_KF_config)
    est_IMU_bias = np.zeros((6, 1))

    # Initialize IMU quantization residuals
    quant_residuals = np.zeros((6, 1))

    # Generate IMU bias and clock output records
    out_IMU_bias_est[0, 0] = old_time
    out_IMU_bias_est[[0], 1:7] = np.transpose(est_IMU_bias)
    out_clock[0, 0] = old_time
    out_clock[[0], 1: 3] = np.transpose(est_clock)

    # Generate KF uncertainty record
    out_KF_SD[0, 0] = old_time

    for i in range(1,18):
        out_KF_SD[0, i] = math.sqrt(P_matrix[i-1, i-1])

    # Initialize GNSS model timing
    time_last_GNSS = old_time
    GNSS_epoch = 0

    # Main loop
    for epoch in range(1, no_epochs):
        # Input time from motion profile
        time = in_profile[epoch, 0]
        true_L_b = in_profile[epoch, 1]
        true_lambda_b = in_profile[epoch, 2]
        true_h_b = in_profile[epoch, 3]
        true_v_eb_n = np.transpose(in_profile[[epoch], 4:7])
        true_eul_nb = np.transpose(in_profile[[epoch], 7:])
        true_C_b_n = np.transpose(euler_to_ctm(true_eul_nb))

        true_r_eb_e, true_v_eb_e, true_C_b_e = ned_to_ecef(true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)

        # Time interval
        tor_i = time - old_time

        # Calculate specific force and angular rate
        true_f_ib_b, true_omega_ib_b = kinematics_ecef(tor_i, true_C_b_e, old_true_C_b_e, true_v_eb_e, old_true_v_eb_e,
                                                       old_true_r_eb_e)

        # Simulate IMU errors
        meas_f_ib_b, meas_omega_ib_b, quant_residuals = imu_model(tor_i, true_f_ib_b, true_omega_ib_b, IMU_errors,
                                                                  quant_residuals)

        # Correct IMU errors
        meas_f_ib_b = meas_f_ib_b - est_IMU_bias[0:3, [0]]
        meas_omega_ib_b = meas_omega_ib_b - est_IMU_bias[3:6, [0]]

        # Update estimated navigation solution
        est_r_eb_e, est_v_eb_e, est_C_b_e = nav_equations_ecef(tor_i, old_est_r_eb_e, old_est_v_eb_e, old_est_C_b_e,
                                                               meas_f_ib_b, meas_omega_ib_b)

        # Determine whether to update GNSS simulation and run Kalman filter
        if (time - time_last_GNSS) >= GNSS_config.epoch_interval:
            GNSS_epoch = GNSS_epoch + 1
            tor_s = time - time_last_GNSS # KF time interval
            time_last_GNSS = time

            # Determine satellite positions and velocities
            sat_r_es_e, sat_v_es_e = satellite_positions_and_velocities(time, GNSS_config)

            # Generate GNSS measurements
            GNSS_measurements, no_GNSS_meas = generate_gnss_measurements(time, sat_r_es_e, sat_v_es_e, true_r_eb_e,
                                                                         true_L_b, true_lambda_b, true_v_eb_e,
                                                                         GNSS_biases, GNSS_config)

            # Run integration kalman filter
            est_C_b_e, est_v_eb_e, est_r_eb_e, est_IMU_bias, est_clock, P_matrix = tc_kf_epoch\
                (GNSS_measurements, no_GNSS_meas, tor_s, est_C_b_e, est_v_eb_e, est_r_eb_e, est_IMU_bias, est_clock, P_matrix,
                 meas_f_ib_b, est_L_b, TC_KF_config)

            # Generate IMU bias and clock output records
            out_IMU_bias_est[GNSS_epoch, 0] = time
            out_IMU_bias_est[[GNSS_epoch], 1:7] = np.transpose(est_IMU_bias)
            out_clock[GNSS_epoch, 0] = time
            out_clock[[GNSS_epoch], 1: 3] = np.transpose(est_clock)

            # Generate KF uncertainty record
            out_KF_SD[GNSS_epoch, 0] = old_time

            for i in range(1, 18):
                out_KF_SD[GNSS_epoch, i] = math.sqrt(P_matrix[i - 1, i - 1])

        # Convert navigation solution to NED
        est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n = ecef_to_ned(est_r_eb_e, est_v_eb_e, est_C_b_e)

        # Generate output profile record
        out_profile[epoch, 0] = time
        out_profile[epoch, 1] = est_L_b
        out_profile[epoch, 2] = est_lambda_b
        out_profile[epoch, 3] = est_h_b
        out_profile[[epoch], 4: 7] = np.transpose(est_v_eb_n)
        out_profile[[epoch], 7:] = np.transpose(ctm_to_euler(np.transpose(est_C_b_n)))

        # Determine errors and generate output record
        delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = calculate_errors_ned(
            est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n, true_L_b,
            true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)
        out_errors[epoch, 0] = time
        out_errors[[epoch], 1: 4] = np.transpose(delta_r_eb_n)
        out_errors[[epoch], 4: 7] = np.transpose(delta_v_eb_n)
        out_errors[[epoch], 7:] = np.transpose(delta_eul_nb_n)

        # Reset old values
        old_time = time
        old_true_r_eb_e = true_r_eb_e
        old_true_v_eb_e = true_v_eb_e
        old_true_C_b_e = true_C_b_e
        old_est_r_eb_e = est_r_eb_e
        old_est_v_eb_e = est_v_eb_e
        old_est_C_b_e = est_C_b_e

    return out_profile, out_errors, out_IMU_bias_est, out_clock, out_KF_SD


def loosely_coupled_ins_gnss(in_profile, no_epochs, initialization_errors, IMU_errors, GNSS_config, LC_KF_config):
    """
    Loosely_coupled_INS_GNSS - Simulates inertial navigation using ECEF
    navigation equations and kinematic model, GNSS using a least-squares
    positioning algorithm, and loosely-coupled INS/GNSS integration.

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 12/4/2012 by Paul Groves

    Inputs:
      in_profile   True motion profile array
      no_epochs    Number of epochs of profile data
      initialization_errors
        .delta_r_eb_n     position error resolved along NED (m)
        .delta_v_eb_n     velocity error resolved along NED (m/s)
        .delta_eul_nb_n   attitude error as NED Euler angles (rad)
      IMU_errors
        .delta_r_eb_n     position error resolved along NED (m)
        .b_a              Accelerometer biases (m/s^2)
        .b_g              Gyro biases (rad/s)
        .M_a              Accelerometer scale factor and cross coupling errors
        .M_g              Gyro scale factor and cross coupling errors
        .G_g              Gyro g-dependent biases (rad-sec/m)
        .accel_noise_root_PSD   Accelerometer noise root PSD (m s^-1.5)
        .gyro_noise_root_PSD    Gyro noise root PSD (rad s^-0.5)
        .accel_quant_level      Accelerometer quantization level (m/s^2)
        .gyro_quant_level       Gyro quantization level (rad/s)
      GNSS_config
        .epoch_interval     Interval between GNSS epochs (s)
        .init_est_r_ea_e    Initial estimated position (m; ECEF)
        .no_sat             Number of satellites in constellation
        .r_os               Orbital radius of satellites (m)
        .inclination        Inclination angle of satellites (deg)
        .const_delta_lambda Longitude offset of constellation (deg)
        .const_delta_t      Timing offset of constellation (s)
        .mask_angle         Mask angle (deg)
        .SIS_err_SD         Signal in space error SD (m)
        .zenith_iono_err_SD Zenith ionosphere error SD (m)
        .zenith_trop_err_SD Zenith troposphere error SD (m)
        .code_track_err_SD  Code tracking error SD (m)
        .rate_track_err_SD  Range rate tracking error SD (m/s)
        .rx_clock_offset    Receiver clock offset at time=0 (m)
        .rx_clock_drift     Receiver clock drift at time=0 (m/s)
      LC_KF_config
        .init_att_unc           Initial attitude uncertainty per axis (rad)
        .init_vel_unc           Initial velocity uncertainty per axis (m/s)
        .init_pos_unc           Initial position uncertainty per axis (m)
        .init_b_a_unc           Initial accel. bias uncertainty (m/s^2)
        .init_b_g_unc           Initial gyro. bias uncertainty (rad/s)
        .gyro_noise_PSD         Gyro noise PSD (rad^2/s)
        .accel_noise_PSD        Accelerometer noise PSD (m^2 s^-3)
        .accel_bias_PSD         Accelerometer bias random walk PSD (m^2 s^-5)
        .gyro_bias_PSD          Gyro bias random walk PSD (rad^2 s^-3)
        .pos_meas_SD            Position measurement noise SD per axis (m)
        .vel_meas_SD            Velocity measurement noise SD per axis (m/s)

    Outputs:
      out_profile        Navigation solution as a motion profile array
      out_errors         Navigation solution error array
      out_IMU_bias_est   Kalman filter IMU bias estimate array
      out_clock          GNSS Receiver clock estimate array
      out_KF_SD          Output Kalman filter state uncertainties

    Format of motion profiles:
     Column 1: time (sec)
     Column 2: latitude (rad)
     Column 3: longitude (rad)
     Column 4: height (m)
     Column 5: north velocity (m/s)
     Column 6: east velocity (m/s)
     Column 7: down velocity (m/s)
     Column 8: roll angle of body w.r.t NED (rad)
     Column 9: pitch angle of body w.r.t NED (rad)
     Column 10: yaw angle of body w.r.t NED (rad)

    Format of error array:
     Column 1: time (sec)
     Column 2: north position error (m)
     Column 3: east position error (m)
     Column 4: down position error (m)
     Column 5: north velocity error (m/s)
     Column 6: east velocity error (m/s)
     Column 7: down velocity (error m/s)
     Column 8: attitude error about north (rad)
     Column 9: attitude error about east (rad)
     Column 10: attitude error about down = heading error  (rad)

    Format of output IMU biases array:
     Column 1: time (sec)
     Column 2: estimated X accelerometer bias (m/s^2)
     Column 3: estimated Y accelerometer bias (m/s^2)
     Column 4: estimated Z accelerometer bias (m/s^2)
     Column 5: estimated X gyro bias (rad/s)
     Column 6: estimated Y gyro bias (rad/s)
     Column 7: estimated Z gyro bias (rad/s)

    Format of receiver clock array:
     Column 1: time (sec)
     Column 2: estimated clock offset (m)
     Column 3: estimated clock drift (m/s)

    Format of KF state uncertainties array:
     Column 1: time (sec)
     Column 2: X attitude error uncertainty (rad)
     Column 3: Y attitude error uncertainty (rad)
     Column 4: Z attitude error uncertainty (rad)
     Column 5: X velocity error uncertainty (m/s)
     Column 6: Y velocity error uncertainty (m/s)
     Column 7: Z velocity error uncertainty (m/s)
     Column 8: X position error uncertainty (m)
     Column 9: Y position error uncertainty (m)
     Column 10: Z position error uncertainty (m)
     Column 11: X accelerometer bias uncertainty (m/s^2)
     Column 12: Y accelerometer bias uncertainty (m/s^2)
     Column 13: Z accelerometer bias uncertainty (m/s^2)
     Column 14: X gyro bias uncertainty (rad/s)
     Column 15: Y gyro bias uncertainty (rad/s)
     Column 16: Z gyro bias uncertainty (rad/s)
     Column 17: clock offset uncertainty (m)
     Column 18: clock drift uncertainty (m/s)

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    # Initialize true navigation solution
    old_time = in_profile[0, 0]                              # time (sec)
    true_L_b = in_profile[0, 1]                             # latitude (rad)
    true_lambda_b = in_profile[0, 2]                        # longitude (rad)
    true_h_b = in_profile[0, 3]
    true_v_eb_n = np.transpose(in_profile[[0], 4:7])        # velocity of body frame wrt ECEF resolved in NED frame (m/s)
    true_eul_nb = np.transpose(in_profile[[0], 7:])         # euler angle of body frame wrt NED frame (rad)
    true_C_b_n = np.transpose(euler_to_ctm(true_eul_nb))    # rotation matrix from body to NED frame

    old_true_r_eb_e, old_true_v_eb_e, old_true_C_b_e = ned_to_ecef(true_L_b, true_lambda_b, true_h_b, true_v_eb_n,
                                                                   true_C_b_n)

    # Determine satellite positions and velocities
    sat_r_es_e, sat_v_es_e = satellite_positions_and_velocities(old_time, GNSS_config)

    # Initialize the GNSS biases. Note that these are assumed constant throughout
    # the simulation and are based on the initial elevation angles.Therefore,
    # this function is unsuited to simulations longer than about 30 min.
    GNSS_biases = initialize_gnss_biases(sat_r_es_e, old_true_r_eb_e, true_L_b, true_lambda_b, GNSS_config)

    # Generate GNSS measurements
    GNSS_measurements, no_GNSS_meas = generate_gnss_measurements(old_time, sat_r_es_e, sat_v_es_e, old_true_r_eb_e,
                                                                 true_L_b, true_lambda_b, old_true_v_eb_e,
                                                                 GNSS_biases, GNSS_config)

    # Determine Least-squares GNSS position solution
    GNSS_r_eb_e, GNSS_v_eb_e, est_clock = gnss_ls_position_velocity(GNSS_measurements, no_GNSS_meas,
                                                                          GNSS_config.init_est_r_ea_e, np.zeros((3, 1)))
    old_est_r_eb_e = GNSS_r_eb_e
    old_est_v_eb_e = GNSS_v_eb_e
    old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n = pv_ecef_to_ned(old_est_r_eb_e, old_est_v_eb_e)
    est_L_b = old_est_L_b

    # Initialize estimated attitude solution
    old_est_C_b_n = initialize_ned_attitude(true_C_b_n, initialization_errors)
    temp1, temp2, old_est_C_b_e = ned_to_ecef(old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_n)

    # Initialize output profile record and errors record
    out_profile = np.zeros((no_epochs, 10))
    out_errors = np.zeros((no_epochs, 10))
    out_clock = np.zeros((no_epochs, 3))
    out_KF_SD = np.zeros((no_epochs, 18))
    out_IMU_bias_est = np.zeros((no_epochs, 7))

    # Generate output profile record
    out_profile[0, 0] = old_time
    out_profile[0, 1] = old_est_L_b
    out_profile[0, 2] = old_est_lambda_b
    out_profile[0, 3] = old_est_h_b
    out_profile[[0], 4: 7] = np.transpose(old_est_v_eb_n)
    out_profile[[0], 7:] = np.transpose(ctm_to_euler(np.transpose(old_est_C_b_n)))

    # Determine errors and generate output record
    delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = calculate_errors_ned(
        old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_n, true_L_b,
        true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)
    out_errors[0, 0] = old_time
    out_errors[[0], 1: 4] = np.transpose(delta_r_eb_n)
    out_errors[[0], 4: 7] = np.transpose(delta_v_eb_n)
    out_errors[[0], 7:] = np.transpose(delta_eul_nb_n)

    # Initialize Kalman filter P matrix and IMU bias states
    P_matrix = initialize_lc_P_matrix(LC_KF_config)
    est_IMU_bias = np.zeros((6, 1))

    # Initialize IMU quantization residuals
    quant_residuals = np.zeros((6, 1))

    # Generate IMU bias and clock output records
    out_IMU_bias_est[0, 0] = old_time
    out_IMU_bias_est[[0], 1:7] = np.transpose(est_IMU_bias)
    out_clock[0, 0] = old_time
    out_clock[[0], 1: 3] = np.transpose(est_clock)

    # Generate KF uncertainty record
    out_KF_SD[0, 0] = old_time

    for i in range(1, 16):
        out_KF_SD[0, i] = math.sqrt(P_matrix[i - 1, i - 1])

    # Initialize GNSS model timing
    time_last_GNSS = old_time
    GNSS_epoch = 0

    # Main loop
    for epoch in range(1, no_epochs):
        # Input time from motion profile
        time = in_profile[epoch, 0]
        true_L_b = in_profile[epoch, 1]
        true_lambda_b = in_profile[epoch, 2]
        true_h_b = in_profile[epoch, 3]
        true_v_eb_n = np.transpose(in_profile[[epoch], 4:7])
        true_eul_nb = np.transpose(in_profile[[epoch], 7:])
        true_C_b_n = np.transpose(euler_to_ctm(true_eul_nb))

        true_r_eb_e, true_v_eb_e, true_C_b_e = ned_to_ecef(true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)

        # Time interval
        tor_i = time - old_time

        # Calculate specific force and angular rate
        true_f_ib_b, true_omega_ib_b = kinematics_ecef(tor_i, true_C_b_e, old_true_C_b_e, true_v_eb_e, old_true_v_eb_e,
                                                       old_true_r_eb_e)

        # Simulate IMU errors
        meas_f_ib_b, meas_omega_ib_b, quant_residuals = imu_model(tor_i, true_f_ib_b, true_omega_ib_b, IMU_errors,
                                                                  quant_residuals)

        # Correct IMU errors
        meas_f_ib_b = meas_f_ib_b - est_IMU_bias[0:3, [0]]
        meas_omega_ib_b = meas_omega_ib_b - est_IMU_bias[3:6, [0]]

        # Update estimated navigation solution
        est_r_eb_e, est_v_eb_e, est_C_b_e = nav_equations_ecef(tor_i, old_est_r_eb_e, old_est_v_eb_e, old_est_C_b_e,
                                                               meas_f_ib_b, meas_omega_ib_b)

        # Determine whether to update GNSS simulation and run Kalman filter
        if (time - time_last_GNSS) >= GNSS_config.epoch_interval:
            GNSS_epoch = GNSS_epoch + 1
            tor_s = time - time_last_GNSS  # KF time interval
            time_last_GNSS = time

            # Determine satellite positions and velocities
            sat_r_es_e, sat_v_es_e = satellite_positions_and_velocities(time, GNSS_config)

            # Generate GNSS measurements
            GNSS_measurements, no_GNSS_meas = generate_gnss_measurements(time, sat_r_es_e, sat_v_es_e, true_r_eb_e,
                                                                         true_L_b, true_lambda_b, true_v_eb_e,
                                                                         GNSS_biases, GNSS_config)

            # Determine GNSS position solution
            GNSS_r_eb_e, GNSS_v_eb_e, est_clock = gnss_ls_position_velocity(GNSS_measurements, no_GNSS_meas, GNSS_r_eb_e, GNSS_v_eb_e)

            # Run integration kalman filter
            est_C_b_e, est_v_eb_e, est_r_eb_e, est_IMU_bias, P_matrix = lc_kf_epoch \
                (GNSS_r_eb_e, GNSS_v_eb_e, tor_s, est_C_b_e, est_v_eb_e, est_r_eb_e, est_IMU_bias, P_matrix,
                 meas_f_ib_b, est_L_b, LC_KF_config)

            # Generate IMU bias and clock output records
            out_IMU_bias_est[GNSS_epoch, 0] = time
            out_IMU_bias_est[[GNSS_epoch], 1:7] = np.transpose(est_IMU_bias)
            out_clock[GNSS_epoch, 0] = time
            out_clock[[GNSS_epoch], 1: 3] = np.transpose(est_clock)

            # Generate KF uncertainty record
            out_KF_SD[GNSS_epoch, 0] = old_time

            for i in range(1, 16):
                out_KF_SD[GNSS_epoch, i] = math.sqrt(P_matrix[i - 1, i - 1])

        # Convert navigation solution to NED
        est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n = ecef_to_ned(est_r_eb_e, est_v_eb_e, est_C_b_e)

        # Generate output profile record
        out_profile[epoch, 0] = time
        out_profile[epoch, 1] = est_L_b
        out_profile[epoch, 2] = est_lambda_b
        out_profile[epoch, 3] = est_h_b
        out_profile[[epoch], 4: 7] = np.transpose(est_v_eb_n)
        out_profile[[epoch], 7:] = np.transpose(ctm_to_euler(np.transpose(est_C_b_n)))

        # Determine errors and generate output record
        delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = calculate_errors_ned(
            est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n, true_L_b,
            true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)
        out_errors[epoch, 0] = time
        out_errors[epoch, 1: 4] = np.transpose(delta_r_eb_n)
        out_errors[[epoch], 4: 7] = np.transpose(delta_v_eb_n)
        out_errors[[epoch], 7:] = np.transpose(delta_eul_nb_n)

        # Reset old values
        old_time = time
        old_true_r_eb_e = true_r_eb_e
        old_true_v_eb_e = true_v_eb_e
        old_true_C_b_e = true_C_b_e
        old_est_r_eb_e = est_r_eb_e
        old_est_v_eb_e = est_v_eb_e
        old_est_C_b_e = est_C_b_e

    return out_profile, out_errors, out_IMU_bias_est, out_clock, out_KF_SD
