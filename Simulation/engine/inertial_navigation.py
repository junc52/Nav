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
from Simulation.engine.kinematics import *
from Simulation.engine.initialize_models import initialize_ned, imu_model
from Simulation.engine.nav_equations import *
from Simulation.engine.calculate_errors import calculate_errors_ned


def inertial_navigation_eci(in_profile, no_epochs, initialization_errors, IMU_errors):
    """
    Inertial_navigation_ECI - Simulates inertial navigation using ECI
    navigation equations and kinematic model

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 3/4/2012 by Paul Groves

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

    Outputs:
        out_profile   Navigation solution as a motion profile array
        out_errors    Navigation solution error array

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
    Column 5: north velocity (m/s)
    Column 6: east velocity (m/s)
    Column 7: down velocity (m/s)
    Column 8: attitude error about north (rad)
    Column 9: attitude error about east (rad)
    Column 10: attitude error about down = heading error  (rad)

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    # Initialize true navigation solution
    old_time      = in_profile[0, 0]                                        # time (sec)
    true_L_b      = in_profile[0, 1]                                        # latitude (rad)
    true_lambda_b = in_profile[0, 2]                                        # longitude (rad)
    true_h_b      = in_profile[0, 3]
    true_v_eb_n   = np.transpose(in_profile[[0], 4:7])                      # velocity of body frame wrt ECEF resolved in NED frame (m/s)
    true_eul_nb   = np.transpose(in_profile[[0], 7:])                       # euler angle of body frame wrt NED frame (rad)
    true_C_b_n    = np.transpose(euler_to_ctm(true_eul_nb))                 # rotation matrix from body to NED frame

    true_r_eb_e, true_v_eb_e, true_C_b_e = ned_to_ecef(true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)
    old_true_r_ib_i, old_true_v_ib_i, old_true_C_b_i = ecef_to_eci(old_time, true_r_eb_e, true_v_eb_e, true_C_b_e)

    # Initialize estimated navigation solution
    old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_n = \
        initialize_ned(true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n, initialization_errors)

    # Initialize output profile record and errors record
    out_profile = np.zeros((no_epochs, 10))
    out_errors = np.zeros((no_epochs, 10))

    # Generate output profile record
    out_profile[0, 0]    = old_time
    out_profile[0, 1]    = old_est_L_b
    out_profile[0, 2]    = old_est_lambda_b
    out_profile[0, 3]    = old_est_h_b
    out_profile[[0], 4: 7] = np.transpose(old_est_v_eb_n)
    out_profile[[0], 7:]   = np.transpose(ctm_to_euler(np.transpose(old_est_C_b_n)))

    out_errors[0, 0]    = old_time
    out_errors[[0], 1: 4] = np.transpose(initialization_errors.delta_r_eb_n)
    out_errors[[0], 4: 7] = np.transpose(initialization_errors.delta_v_eb_n)
    out_errors[[0], 7:]   = np.transpose(initialization_errors.delta_eul_nb_n)
    [old_est_r_eb_e, old_est_v_eb_e, old_est_C_b_e] = ned_to_ecef(old_est_L_b,
                                                                  old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_n)
    [old_est_r_ib_i, old_est_v_ib_i, old_est_C_b_i] = ecef_to_eci(old_time,
                                                                  old_est_r_eb_e, old_est_v_eb_e, old_est_C_b_e)
    # Initialize IMU quantization residuals
    quant_residuals = np.array([[0], [0], [0], [0], [0], [0]])

    # Main loop
    for epoch in range(1, no_epochs):
        # Input data from motion profile
        time = in_profile[epoch, 0]
        true_L_b = in_profile[epoch, 1]
        true_lambda_b = in_profile[epoch, 2]
        true_h_b = in_profile[epoch, 3]
        true_v_eb_n = np.transpose(in_profile[[epoch], 4:7])
        true_eul_nb = np.transpose(in_profile[[epoch], 7:])
        true_C_b_n = np.transpose(euler_to_ctm(true_eul_nb))

        true_r_eb_e, true_v_eb_e, true_C_b_e = ned_to_ecef(true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)
        true_r_ib_i, true_v_ib_i, true_C_b_i = ecef_to_eci(time, true_r_eb_e, true_v_eb_e, true_C_b_e)

        # Time interval
        tor_i = time - old_time

        # Calculate specific force and angular rate
        true_f_ib_b, true_omega_ib_b = kinematics_eci(tor_i, true_C_b_i, old_true_C_b_i, true_v_ib_i, old_true_v_ib_i, old_true_r_ib_i)

        # Simulate IMU errors
        meas_f_ib_b, meas_omega_ib_b, quant_residuals = imu_model(tor_i, true_f_ib_b, true_omega_ib_b, IMU_errors, quant_residuals)

        # Update estimated navigation solution
        est_r_ib_i, est_v_ib_i, est_C_b_i = nav_equations_eci(tor_i, old_est_r_ib_i, old_est_v_ib_i, old_est_C_b_i,
                                                              meas_f_ib_b, meas_omega_ib_b)

        est_r_eb_e, est_v_eb_e, est_C_b_e = eci_to_ecef(time, est_r_ib_i, est_v_ib_i, est_C_b_i)
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
        old_true_r_ib_i = true_r_ib_i
        old_true_v_ib_i = true_v_ib_i
        old_true_C_b_i = true_C_b_i
        old_est_r_ib_i = est_r_ib_i
        old_est_v_ib_i = est_v_ib_i
        old_est_C_b_i = est_C_b_i

    return out_profile, out_errors


def inertial_navigation_ecef(in_profile, no_epochs, initialization_errors, IMU_errors):
    """
    Inertial_navigation_ECEF - Simulates inertial navigation using ECEF
    navigation equations and kinematic model

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 3/4/2012 by Paul Groves

    Input, Output, and motion profiles is same as eci version
    """

    # Initialize true navigation solution
    old_time = in_profile[0, 0]                             # time (sec)
    true_L_b = in_profile[0, 1]                             # latitude (rad)
    true_lambda_b = in_profile[0, 2]                        # longitude (rad)
    true_h_b = in_profile[0, 3]
    true_v_eb_n = np.transpose(in_profile[[0], 4:7])          # velocity of body frame wrt ECEF resolved in NED frame (m/s)
    true_eul_nb = np.transpose(in_profile[[0], 7:])           # euler angle of body frame wrt NED frame (rad)
    true_C_b_n = np.transpose(euler_to_ctm(true_eul_nb))    # rotation matrix from body to NED frame
    old_true_r_eb_e, old_true_v_eb_e, old_true_C_b_e = ned_to_ecef(true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)

    # Initialize estimated navigation solution
    old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_n = \
        initialize_ned(true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n, initialization_errors)

    # Initialize output profile record and errors record
    out_profile = np.zeros((no_epochs, 10))
    out_errors = np.zeros((no_epochs, 10))

    # Generate output profile record
    out_profile[0, 0] = old_time
    out_profile[0, 1] = old_est_L_b
    out_profile[0, 2] = old_est_lambda_b
    out_profile[0, 3] = old_est_h_b
    out_profile[[0], 4: 7] = np.transpose(old_est_v_eb_n)
    out_profile[[0], 7:] = np.transpose(ctm_to_euler(np.transpose(old_est_C_b_n)))

    out_errors[0, 0] = old_time
    out_errors[[0], 1: 4] = np.transpose(initialization_errors.delta_r_eb_n)
    out_errors[[0], 4: 7] = np.transpose(initialization_errors.delta_v_eb_n)
    out_errors[[0], 7:] = np.transpose(initialization_errors.delta_eul_nb_n)
    [old_est_r_eb_e, old_est_v_eb_e, old_est_C_b_e] = ned_to_ecef(old_est_L_b,
                                                                  old_est_lambda_b, old_est_h_b, old_est_v_eb_n,
                                                                  old_est_C_b_n)

    # Initialize IMU quantization residuals
    quant_residuals = np.array([[0], [0], [0], [0], [0], [0]])

    # Main loop
    for epoch in range(1, no_epochs):
        # Input data from motion profile
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

        # Update estimated navigation solution
        est_r_eb_e, est_v_eb_e, est_C_b_e = nav_equations_ecef(tor_i, old_est_r_eb_e, old_est_v_eb_e, old_est_C_b_e,
                                                              meas_f_ib_b, meas_omega_ib_b)

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

    return out_profile, out_errors


def inertial_navigation_ned(in_profile, no_epochs, initialization_errors, IMU_errors):
    """
    Inertial_navigation_NED - Simulates inertial navigation using local
    navigation frame (NED) navigation equations and kinematic model

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 3/4/2012 by Paul Groves

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

    Outputs:
      out_profile   Navigation solution as a motion profile array
      out_errors    Navigation solution error array

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
     Column 5: north velocity (m/s)
     Column 6: east velocity (m/s)
     Column 7: down velocity (m/s)
     Column 8: attitude error about north (rad)
     Column 9: attitude error about east (rad)
     Column 10: attitude error about down = heading error  (rad)

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """

    # Initialize true navigation solution
    old_time = in_profile[0, 0]                                     # time (sec)
    old_true_L_b = in_profile[0, 1]                                 # latitude (rad)
    old_true_lambda_b = in_profile[0, 2]                            # longitude (rad)
    old_true_h_b = in_profile[0, 3]
    old_true_v_eb_n = np.transpose(in_profile[[0], 4:7])            # velocity of body frame wrt ECEF resolved in NED frame (m/s)
    old_true_eul_nb = np.transpose(in_profile[[0], 7:])             # euler angle of body frame wrt NED frame (rad)
    old_true_C_b_n = np.transpose(euler_to_ctm(old_true_eul_nb))    # rotation matrix from body to NED frame

    # Initialize estimated navigation solution
    old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_n = \
        initialize_ned(old_true_L_b, old_true_lambda_b, old_true_h_b, old_true_v_eb_n, old_true_C_b_n, initialization_errors)

    # Initialize output profile record and errors record
    out_profile = np.zeros((no_epochs, 10))
    out_errors = np.zeros((no_epochs, 10))

    # Generate output profile record
    out_profile[0, 0] = old_time
    out_profile[0, 1] = old_est_L_b
    out_profile[0, 2] = old_est_lambda_b
    out_profile[0, 3] = old_est_h_b
    out_profile[[0], 4: 7] = np.transpose(old_est_v_eb_n)
    out_profile[[0], 7:] = np.transpose(ctm_to_euler(np.transpose(old_est_C_b_n)))

    out_errors[0, 0] = old_time
    out_errors[[0], 1: 4] = np.transpose(initialization_errors.delta_r_eb_n)
    out_errors[[0], 4: 7] = np.transpose(initialization_errors.delta_v_eb_n)
    out_errors[[0], 7:] = np.transpose(initialization_errors.delta_eul_nb_n)

    # Initialize IMU quantization residuals
    quant_residuals = np.array([[0], [0], [0], [0], [0], [0]])

    # Main loop
    for epoch in range(1, no_epochs):
        # Input data from motion profile
        time = in_profile[epoch, 0]
        true_L_b = in_profile[epoch, 1]
        true_lambda_b = in_profile[epoch, 2]
        true_h_b = in_profile[epoch, 3]
        true_v_eb_n = np.transpose(in_profile[[epoch], 4:7])
        true_eul_nb = np.transpose(in_profile[[epoch], 7:])
        true_C_b_n = np.transpose(euler_to_ctm(true_eul_nb))

        # Time interval
        tor_i = time - old_time

        # Calculate specific force and angular rate
        true_f_ib_b, true_omega_ib_b = kinematics_ned(tor_i, true_C_b_n, old_true_C_b_n, true_v_eb_n, old_true_v_eb_n,
                                                      true_L_b, true_h_b, old_true_L_b, old_true_h_b)

        # Simulate IMU errors
        meas_f_ib_b, meas_omega_ib_b, quant_residuals = imu_model(tor_i, true_f_ib_b, true_omega_ib_b, IMU_errors,
                                                                  quant_residuals)

        # Update estimated navigation solution
        est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n = nav_equations_ned(tor_i, old_est_L_b, old_est_lambda_b, old_est_h_b,
                                                              old_est_v_eb_n, old_est_C_b_n, meas_f_ib_b, meas_omega_ib_b)


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
        old_true_L_b = true_L_b
        old_true_lambda_b = true_lambda_b
        old_true_h_b = true_h_b
        old_true_v_eb_n = true_v_eb_n
        old_true_C_b_n = true_C_b_n
        old_est_L_b = est_L_b
        old_est_lambda_b = est_lambda_b
        old_est_h_b = est_h_b
        old_est_v_eb_n = est_v_eb_n
        old_est_C_b_n = est_C_b_n

    return out_profile, out_errors


