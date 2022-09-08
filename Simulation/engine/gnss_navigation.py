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
from Simulation.engine.initialize import initialize_gnss_kf
from Simulation.engine.initialize_models import *
from Simulation.engine.gnss_ls_position_velocity import gnss_ls_position_velocity
from Simulation.engine.kalman_filter import gnss_kf_epoch


def gnss_least_squares(in_profile, no_epochs, GNSS_config):
    """
    GNSS_Least_Squares - Simulates stand-alone GNSS using a least-squares
    positioning algorithm

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves

    Inputs:
       in_profile   True motion profile array
       no_epochs    Number of epochs of profile data
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

    Outputs:
       out_profile   Navigation solution as a motion profile array
       out_errors    Navigation solution error array
       out_clock     Receiver clock estimate array

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
      Column 7: down velocity error (m/s)
      Column 8: NOT USED (attitude error about north (rad))
      Column 9: NOT USED (attitude error about east (rad))
      Column 10: NOT USED (attitude error about down = heading error  (rad))

    Format of receiver clock array:
      Column 1: time (sec)
      Column 2: estimated clock offset (m)
      Column 3: estimated clock drift (m/s)

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    # Initialize true navigation solution
    old_time = in_profile[0, 0]                             # time (sec)
    true_L_b = in_profile[0, 1]                             # latitude (rad)
    true_lambda_b = in_profile[0, 2]                        # longitude (rad)
    true_h_b = in_profile[0, 3]
    true_v_eb_n = np.transpose(in_profile[[0], 4:7])        # velocity of body frame wrt ECEF resolved in NED frame (m/s)
    true_eul_nb = np.transpose(in_profile[[0], 7:])         # euler angle of body frame wrt NED frame (rad)
    true_C_b_n = np.transpose(euler_to_ctm(true_eul_nb))    # rotation matrix from body to NED frame

    true_r_eb_e, true_v_eb_e = pv_ned_to_ecef(true_L_b, true_lambda_b, true_h_b, true_v_eb_n)

    time_last_GNSS = old_time
    GNSS_epoch = 0

    # Determine satellite positions and velocities
    sat_r_es_e, sat_v_es_e = satellite_positions_and_velocities(old_time, GNSS_config)

    # Initialize the GNSS biases. Note that these are assumed constant throughout
    # the simulation and are based on the initial elevation angles.Therefore,
    # this function is unsuited to simulations longer than about 30 min.
    GNSS_biases = initialize_gnss_biases(sat_r_es_e, true_r_eb_e, true_L_b, true_lambda_b, GNSS_config)

    # Generate GNSS measurements
    GNSS_measurements, no_GNSS_meas = generate_gnss_measurements(old_time, sat_r_es_e, sat_v_es_e, true_r_eb_e,
                                                                 true_L_b, true_lambda_b, true_v_eb_e,
                                                                 GNSS_biases, GNSS_config)

    # Determine GNSS position solution
    est_r_eb_e, est_v_eb_e, est_clock = gnss_ls_position_velocity(GNSS_measurements, no_GNSS_meas,
                                                                  GNSS_config.init_est_r_ea_e, np.zeros((3,1)))

    est_C_b_n = true_C_b_n  # This sets the attitude errors to zero

    est_L_b, est_lambda_b, est_h_b, est_v_eb_n = pv_ecef_to_ned(est_r_eb_e, est_v_eb_e)

    # Initialize output profile record and errors record
    out_profile = np.zeros((no_epochs, 10))
    out_errors = np.zeros((no_epochs, 10))
    out_clock = np.zeros((no_epochs, 3))

    # Generate output profile record
    out_profile[0, 0] = old_time
    out_profile[0, 1] = est_L_b
    out_profile[0, 2] = est_lambda_b
    out_profile[0, 3] = est_h_b
    out_profile[[0], 4: 7] = np.transpose(est_v_eb_n)
    out_profile[[0], 7:] = np.transpose(ctm_to_euler(np.transpose(est_C_b_n)))

    # Determine errors and generate output record
    delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = calculate_errors_ned(
        est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n, true_L_b,
        true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)
    out_errors[0, 0] = old_time
    out_errors[[0], 1: 4] = np.transpose(delta_r_eb_n)
    out_errors[[0], 4: 7] = np.transpose(delta_v_eb_n)
    out_errors[[0], 7:] = np.zeros((1, 3))

    # Generate clock output record
    out_clock[0, 0] = old_time
    out_clock[[0], 1: 3] = np.transpose(est_clock[:, [0]])

    # Main loop
    for epoch in range(1, no_epochs):
        # Input time from motion profile
        time = in_profile[epoch, 0]

        # Determine whether to update GNSS simulation
        if (time - time_last_GNSS) >= GNSS_config.epoch_interval:
            GNSS_epoch = GNSS_epoch + 1
            time_last_GNSS = time

            # Input data from motion profile
            true_L_b = in_profile[epoch, 1]
            true_lambda_b = in_profile[epoch, 2]
            true_h_b = in_profile[epoch, 3]
            true_v_eb_n = np.transpose(in_profile[[epoch], 4:7])
            true_eul_nb = np.transpose(in_profile[[epoch], 7:])
            true_C_b_n = np.transpose(euler_to_ctm(true_eul_nb))

            true_r_eb_e, true_v_eb_e = pv_ned_to_ecef(true_L_b, true_lambda_b, true_h_b, true_v_eb_n)

            # Determine satellite positions and velocities
            sat_r_es_e, sat_v_es_e = satellite_positions_and_velocities(time, GNSS_config)

            # Generate GNSS measurements
            GNSS_measurements, no_GNSS_meas = generate_gnss_measurements(time, sat_r_es_e, sat_v_es_e, true_r_eb_e,
                                                                         true_L_b, true_lambda_b, true_v_eb_e,
                                                                         GNSS_biases, GNSS_config)

            # Determine GNSS position solution
            est_r_eb_e, est_v_eb_e, est_clock = gnss_ls_position_velocity(GNSS_measurements, no_GNSS_meas,
                                                                          est_r_eb_e, est_v_eb_e)

            est_L_b, est_lambda_b, est_h_b, est_v_eb_n = pv_ecef_to_ned(est_r_eb_e, est_v_eb_e)

            est_C_b_n = true_C_b_n  # This sets the attitude errors to zero

            # Generate output profile record
            out_profile[GNSS_epoch, 0] = time
            out_profile[GNSS_epoch, 1] = est_L_b
            out_profile[GNSS_epoch, 2] = est_lambda_b
            out_profile[GNSS_epoch, 3] = est_h_b
            out_profile[[GNSS_epoch], 4: 7] = np.transpose(est_v_eb_n)
            out_profile[[GNSS_epoch], 7:] = np.transpose(ctm_to_euler(np.transpose(est_C_b_n)))

            # Determine errors and generate output record
            delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = calculate_errors_ned(
                est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n, true_L_b,
                true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)
            out_errors[[GNSS_epoch], 0] = time
            out_errors[[GNSS_epoch], 1: 4] = np.transpose(delta_r_eb_n)
            out_errors[[GNSS_epoch], 4: 7] = np.transpose(delta_v_eb_n)
            out_errors[[GNSS_epoch], 7:] = np.zeros((1, 3))

            # Generate clock output record
            out_clock[GNSS_epoch, 0] = time
            out_clock[[GNSS_epoch], 1: 3] = np.transpose(est_clock[:, [0]])

            # Reset old values
            old_time = time

    return out_profile[:GNSS_epoch, :], out_errors[:GNSS_epoch, :], out_clock[:GNSS_epoch, :]


def gnss_kalman_filter(in_profile, no_epochs, GNSS_config, GNSS_KF_config):
    """
    GNSS_Kalman_Filter - Simulates stand-alone GNSS using an Extended Kalman
    filter positioning algorithm

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves

    Inputs:
      in_profile   True motion profile array
      no_epochs    Number of epochs of profile data
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
      GNSS_KF_config
        .init_pos_unc           Initial position uncertainty per axis (m)
        .init_vel_unc           Initial velocity uncertainty per axis (m/s)
        .init_clock_offset_unc  Initial clock offset uncertainty per axis (m)
        .init_clock_drift_unc   Initial clock drift uncertainty per axis (m/s)
        .accel_PSD              Acceleration PSD per axis (m^2/s^3)
        .clock_freq_PSD         Receiver clock frequency-drift PSD (m^2/s^3)
        .clock_phase_PSD        Receiver clock phase-drift PSD (m^2/s)
        .pseudo_range_SD        Pseudo-range measurement noise SD (m)
        .range_rate_SD          Pseudo-range rate measurement noise SD (m/s)

    Outputs:
      out_profile   Navigation solution as a motion profile array
      out_errors    Navigation solution error array
      out_clock     Receiver clock estimate array
      out_KF_SD     Output Kalman filter state uncertainties

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
     Column 8: NOT USED (attitude error about north (rad))
     Column 9: NOT USED (attitude error about east (rad))
     Column 10: NOT USED (attitude error about down = heading error (rad))

    Format of receiver clock array:
     Column 1: time (sec)
     Column 2: estimated clock offset (m)
     Column 3: estimated clock drift (m/s)

    Format of KF state uncertainties array:
     Column 1: time (sec)
     Column 2: X position uncertainty (m)
     Column 3: Y position uncertainty (m)
     Column 4: Z position uncertainty (m)
     Column 5: X velocity uncertainty (m/s)
     Column 6: Y velocity uncertainty (m/s)
     Column 7: Z velocity uncertainty (m/s)
     Column 8: clock offset uncertainty (m)
     Column 9: clock drift uncertainty (m/s)

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    # Initialize true navigation solution
    old_time = in_profile[0, 0]                             # time (sec)
    true_L_b = in_profile[0, 1]                             # latitude (rad)
    true_lambda_b = in_profile[0, 2]                        # longitude (rad)
    true_h_b = in_profile[0, 3]
    true_v_eb_n = np.transpose(in_profile[[0], 4:7])        # velocity of body frame wrt ECEF resolved in NED frame (m/s)
    true_eul_nb = np.transpose(in_profile[[0], 7:])         # euler angle of body frame wrt NED frame (rad)
    true_C_b_n = np.transpose(euler_to_ctm(true_eul_nb))    # rotation matrix from body to NED frame

    true_r_eb_e, true_v_eb_e = pv_ned_to_ecef(true_L_b, true_lambda_b, true_h_b, true_v_eb_n)

    time_last_GNSS = old_time
    GNSS_epoch = 0

    # Determine satellite positions and velocities
    sat_r_es_e, sat_v_es_e = satellite_positions_and_velocities(old_time, GNSS_config)

    # Initialize the GNSS biases. Note that these are assumed constant throughout
    # the simulation and are based on the initial elevation angles.Therefore,
    # this function is unsuited to simulations longer than about 30 min.
    GNSS_biases = initialize_gnss_biases(sat_r_es_e, true_r_eb_e, true_L_b, true_lambda_b, GNSS_config)

    # Generate GNSS measurements
    GNSS_measurements, no_GNSS_meas = generate_gnss_measurements(old_time, sat_r_es_e, sat_v_es_e, true_r_eb_e,
                                                                 true_L_b, true_lambda_b, true_v_eb_e,
                                                                 GNSS_biases, GNSS_config)

    # Determine Least-squares GNSS position solution
    est_r_eb_e, est_v_eb_e, est_clock = gnss_ls_position_velocity(GNSS_measurements, no_GNSS_meas,
                                                                  GNSS_config.init_est_r_ea_e, np.zeros((3, 1)))

    # Initialize Kalman filter
    x_est, P_matrix = initialize_gnss_kf(est_r_eb_e, est_v_eb_e, est_clock, GNSS_KF_config)

    est_C_b_n = true_C_b_n  # This sets the attitude errors to zero

    est_L_b, est_lambda_b, est_h_b, est_v_eb_n = pv_ecef_to_ned(x_est[0:3, [0]], x_est[3:6, [0]])

    # Initialize output profile record and errors record
    out_profile = np.zeros((no_epochs, 10))
    out_errors = np.zeros((no_epochs, 10))
    out_clock = np.zeros((no_epochs, 3))
    out_KF_SD = np.zeros((no_epochs, 9))

    # Generate output profile record
    out_profile[0, 0] = old_time
    out_profile[0, 1] = est_L_b
    out_profile[0, 2] = est_lambda_b
    out_profile[0, 3] = est_h_b
    out_profile[[0], 4: 7] = np.transpose(est_v_eb_n)
    out_profile[[0], 7:] = np.transpose(ctm_to_euler(np.transpose(est_C_b_n)))

    # Determine errors and generate output record
    delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = calculate_errors_ned(
        est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n, true_L_b,
        true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)
    out_errors[0, 0] = old_time
    out_errors[[0], 1: 4] = np.transpose(delta_r_eb_n)
    out_errors[[0], 4: 7] = np.transpose(delta_v_eb_n)
    out_errors[[0], 7:] = np.zeros((1, 3))

    # Generate clock output record
    out_clock[0, 0] = old_time
    out_clock[[0], 1: 3] = np.transpose(x_est[6: 8, [0]])

    # Generate KF uncertainty record
    out_KF_SD[[0], 0] = old_time
    for i in range(1, 9):
        out_KF_SD[GNSS_epoch, i] = math.sqrt(P_matrix[i-1, i-1])

    # Main loop
    for epoch in range(1, no_epochs):
        # Input time from motion profile
        time = in_profile[epoch, 0]

        # Determine whether to update GNSS simulation
        if (time - time_last_GNSS) >= GNSS_config.epoch_interval:
            GNSS_epoch = GNSS_epoch + 1
            tor_s = time - time_last_GNSS # KF time interval
            time_last_GNSS = time

            # Input data from motion profile
            true_L_b = in_profile[epoch, 1]
            true_lambda_b = in_profile[epoch, 2]
            true_h_b = in_profile[epoch, 3]
            true_v_eb_n = np.transpose(in_profile[[epoch], 4:7])
            true_eul_nb = np.transpose(in_profile[[epoch], 7:])
            true_C_b_n = np.transpose(euler_to_ctm(true_eul_nb))

            true_r_eb_e, true_v_eb_e = pv_ned_to_ecef(true_L_b, true_lambda_b, true_h_b, true_v_eb_n)

            # Determine satellite positions and velocities
            sat_r_es_e, sat_v_es_e = satellite_positions_and_velocities(time, GNSS_config)

            # Generate GNSS measurements
            GNSS_measurements, no_GNSS_meas = generate_gnss_measurements(time, sat_r_es_e, sat_v_es_e, true_r_eb_e,
                                                                         true_L_b, true_lambda_b, true_v_eb_e,
                                                                         GNSS_biases, GNSS_config)

            # Update GNSS position solution
            x_est, P_matrix = gnss_kf_epoch(GNSS_measurements, no_GNSS_meas, tor_s, x_est, P_matrix, GNSS_KF_config)

            est_L_b, est_lambda_b, est_h_b, est_v_eb_n = pv_ecef_to_ned(x_est[0:3, [0]], x_est[3:6, [0]])

            est_C_b_n = true_C_b_n  # This sets the attitude errors to zero

            # Generate output profile record
            out_profile[GNSS_epoch, 0] = time
            out_profile[GNSS_epoch, 1] = est_L_b
            out_profile[GNSS_epoch, 2] = est_lambda_b
            out_profile[GNSS_epoch, 3] = est_h_b
            out_profile[[GNSS_epoch], 4: 7] = np.transpose(est_v_eb_n)
            out_profile[[GNSS_epoch], 7:] = np.transpose(ctm_to_euler(np.transpose(est_C_b_n)))

            # Determine errors and generate output record
            delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = calculate_errors_ned(
                est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n, true_L_b,
                true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)
            out_errors[GNSS_epoch, 0] = time
            out_errors[[GNSS_epoch], 1: 4] = np.transpose(delta_r_eb_n)
            out_errors[[GNSS_epoch], 4: 7] = np.transpose(delta_v_eb_n)
            out_errors[[GNSS_epoch], 7:] = np.zeros((1, 3))

            # Generate clock output record
            out_clock[GNSS_epoch, 0] = time
            out_clock[[GNSS_epoch], 1: 3] = np.transpose(x_est[6: 8, [0]])

            # Generate KF uncertainty record
            out_KF_SD[GNSS_epoch, 0] = old_time
            for i in range(1, 9):
                out_KF_SD[GNSS_epoch, i] = math.sqrt(P_matrix[i-1, i-1])

            # Reset old values
            old_time = time

    return out_profile[:GNSS_epoch, :], out_errors[:GNSS_epoch, :], out_clock[:GNSS_epoch, :], out_KF_SD[:GNSS_epoch, :]
