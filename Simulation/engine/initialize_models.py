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
from Simulation.utils.curvilinear_conversion import *


def initialize_ned(L_b, lambda_b, h_b, v_eb_n, C_b_n, initialization_errors):
    """
    Initialize_NED - Initializes the curvilinear position, velocity, and
    attitude solution by adding errors to the truth.

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 3/4/2012 by Paul Groves

    Inputs:
        L_b           true latitude (rad)
        lambda_b      true longitude (rad)
        h_b           true height (m)
        v_eb_n        true velocity of body frame w.r.t. ECEF frame, resolved
                      along north, east, and down (m/s)
        C_b_n         true body-to-NED coordinate transformation matrix
        initialization_errors
         .delta_r_eb_n     position error resolved along NED (m)
         .delta_v_eb_n     velocity error resolved along NED (m/s)
         .delta_eul_nb_n   attitude error as NED Euler angles (rad)

    Outputs:
        est_L_b       latitude solution (rad)
        est_lambda_b  longitude solution (rad)
        est_h_b       height solution (m)
        est_v_eb_n    velocity solution of body frame w.r.t. ECEF frame,
                      resolved along north, east, and down (m/s)
        est_C_b_n     body-to-NED coordinate transformation matrix solution
    """
    # Position initialization, using (2.119)
    R_N, R_E     = radii_of_curvature(L_b)
    est_L_b      = L_b + initialization_errors.delta_r_eb_n[0, 0] / (R_N + h_b)
    est_lambda_b = lambda_b + initialization_errors.delta_r_eb_n[1, 0] / ((R_E + h_b) * math.cos(L_b))
    est_h_b      = h_b - initialization_errors.delta_r_eb_n[2, 0]

    # Velocity initialization
    est_v_eb_n = v_eb_n + initialization_errors.delta_v_eb_n

    # Attitude initialization, using (5.109) and (5.111)
    delta_C_b_n = euler_to_ctm(-initialization_errors.delta_eul_nb_n)
    est_C_b_n = np.matmul(delta_C_b_n, C_b_n)

    return est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n


def initialize_ned_attitude(C_b_n,initialization_errors):
    """
    Initialize_NED_attitude - Initializes the attitude solution by adding
    errors to the truth.

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 12/4/2012 by Paul Groves

    Inputs:
      C_b_n         true body-to-NED coordinate transformation matrix
      initialization_errors
        .delta_eul_nb_n   attitude error as NED Euler angles (rad)

    Outputs:
      est_C_b_n     body-to-NED coordinate transformation matrix solution

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    # Attitude initialization, using (5.109) and (5.111)

    # Attitude initialization, using (5.109) and (5.111)
    delta_C_b_n = euler_to_ctm(-initialization_errors.delta_eul_nb_n)
    est_C_b_n = np.matmul(delta_C_b_n, C_b_n)

    return est_C_b_n


def imu_model(tor_i, true_f_ib_b, true_omega_ib_b, imu_errors, old_quant_residuals):
    """
    IMU_model - Simulates an inertial measurement unit (IMU body axes used
    throughout this function)

     Software for use with "Principles of GNSS, Inertial, and Multisensor
     Integrated Navigation Systems," Second Edition.

     This function created 3/4/2012 by Paul Groves

     Inputs:
       tor_i            time interval between epochs (s)
       true_f_ib_b      true specific force of body frame w.r.t. ECEF frame, resolved
                        along body-frame axes, averaged over time interval (m/s^2)
       true_omega_ib_b  true angular rate of body frame w.r.t. ECEF frame, resolved
                        about body-frame axes, averaged over time interval (rad/s)
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
       old_quant_residuals  residuals of previous output quantization process

     Outputs:
       meas_f_ib_b      output specific force of body frame w.r.t. ECEF frame, resolved
                        along body-frame axes, averaged over time interval (m/s^2)
       meas_omega_ib_b  output angular rate of body frame w.r.t. ECEF frame, resolved
                        about body-frame axes, averaged over time interval (rad/s)
       quant_residuals  residuals of output quantization process
    """
    # Generate noise
    if tor_i > 0:
        accel_noise = np.random.randn(3, 1) * imu_errors.accel_noise_root_PSD / math.sqrt(tor_i)
        gyro_noise = np.random.randn(3, 1) * imu_errors.gyro_noise_root_PSD / math.sqrt(tor_i)
    else:
        accel_noise = np.zeros((3, 1))
        gyro_noise = np.zeros((3, 1))

    # Calculate accelerometer and gyro outputs using (4.16) and (4.17)
    uq_f_ib_b = imu_errors.b_a + np.matmul(np.eye(3) + imu_errors.M_a, true_f_ib_b) + accel_noise
    uq_omega_ib_b = imu_errors.b_g + np.matmul(np.eye(3) + imu_errors.M_g, true_omega_ib_b) + \
                    np.matmul(imu_errors.G_g, true_f_ib_b) + gyro_noise

    # Quantize accelerometer outputs
    quant_residuals = np.zeros((6, 1))
    if imu_errors.accel_quant_level > 0:
        meas_f_ib_b = imu_errors.accel_quant_level * np.round((uq_f_ib_b + old_quant_residuals[0:3, [0]]) / imu_errors.accel_quant_level)
        quant_residuals[0:3, [0]] = uq_f_ib_b + old_quant_residuals[0:3, [0]] - meas_f_ib_b

    else:
        meas_f_ib_b = uq_f_ib_b
        quant_residuals[0:3, [0]] = np.zeros((3, 1))

    # Quantize gyro outputs
    if imu_errors.gyro_quant_level > 0:
        meas_omega_ib_b = imu_errors.gyro_quant_level * np.round((uq_omega_ib_b + old_quant_residuals[3:, [0]]) / imu_errors.gyro_quant_level)
        quant_residuals[3:, [0]] = uq_omega_ib_b + old_quant_residuals[3:, [0]] - meas_omega_ib_b

    else:
        meas_omega_ib_b = uq_omega_ib_b
        quant_residuals[3:, [0]] = np.zeros((3, 1))

    return meas_f_ib_b, meas_omega_ib_b, quant_residuals


def satellite_positions_and_velocities(time, GNSS_config):
    """
    Satellite_positions_and_velocities - returns ECEF Cartesian positions and
    ECEF velocities of all satellites in the constellation. Simple circular
    orbits with regularly distributed satellites are modeled.

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves

    Inputs:
      time                  Current simulation time(s)
      GNSS_config
        .no_sat             Number of satellites in constellation
        .r_os               Orbital radius of satellites (m)
        .inclination        Inclination angle of satellites (deg)
        .const_delta_lambda Longitude offset of constellation (deg)
        .const_delta_t      Timing offset of constellation (s)
    Outputs:
      sat_r_es_e (no_sat x 3) ECEF satellite position
      sat_v_es_e (no_sat x 3) ECEF satellite velocity

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details"""

    # Constants (some of these could be changed to inputs at a later date)
    mu = 3.986004418E14     # WGS84 Earth gravitational constant (m^3 s^-2)
    omega_ie = 7.292115e-5  # Earth rotation rate in rad/s

    DEG_TO_RAD = 0.01745329252
    RAD_TO_DEG = 1 / DEG_TO_RAD

    # Convert inclination angle to degrees
    inclination = GNSS_config.inclination * DEG_TO_RAD

    # Determine orbital angular rate using (8.8)
    omega_is = math.sqrt(mu / GNSS_config.r_os ** 3)

    # Determine constellation time
    const_time = time + GNSS_config.const_delta_t

    # Loop satellites
    sat_r_es_e = np.zeros((GNSS_config.no_sat, 3))
    sat_v_es_e = np.zeros((GNSS_config.no_sat, 3))
    for j in range(0, GNSS_config.no_sat):
        # Corrected argument of latitude
        u_os_o = 2 * math.pi * j / GNSS_config.no_sat + omega_is * const_time

        # Satellite position in the orbital frame from (8.14)
        r_os_o = GNSS_config.r_os * np.array([[math.cos(u_os_o)], [math.sin(u_os_o)], [0]])

        # Longitude of the ascending node from (8.16)
        Omega = (math.pi * ((j+1) % 6)/3 + (GNSS_config.const_delta_lambda * DEG_TO_RAD)) - omega_ie*const_time

        # ECEF satellite position from (8.19)
        sat_r_es_e[j, 0] = r_os_o[0, 0] * math.cos(Omega) - r_os_o[1, 0] * math.cos(inclination) * math.sin(Omega)
        sat_r_es_e[j, 1] = r_os_o[0, 0] * math.sin(Omega) + r_os_o[1, 0] * math.cos(inclination) * math.cos(Omega)
        sat_r_es_e[j, 2] = r_os_o[1, 0] * math.sin(inclination)

        # Satellite velocity in the orbital frame from (8.24), noting that with
        # a circular orbit r_os_o is constant and the time derivative of u_os_o
        # is omega_is.
        v_os_o = GNSS_config.r_os * omega_is * np.array([[-math.sin(u_os_o)], [math.cos(u_os_o)], [0]])

        # ECEF satellite velocity from (8.27)
        sat_v_es_e[j, 0] = v_os_o[0, 0] * math.cos(Omega) - v_os_o[1, 0] * math.cos(inclination) * math.sin(Omega) + omega_ie * sat_r_es_e[j, 1]
        sat_v_es_e[j, 1] = v_os_o[0, 0] * math.sin(Omega) + v_os_o[1, 0] * math.cos(inclination) * math.cos(Omega) - omega_ie * sat_r_es_e[j, 0]
        sat_v_es_e[j, 2] = v_os_o[1, 0] * math.sin(inclination)

    return sat_r_es_e, sat_v_es_e


def initialize_gnss_biases(sat_r_es_e, r_ea_e, L_a, lambda_a, GNSS_config):
    """
    Initialize_GNSS_biases - Initializes the GNSS range errors due to signal
    in space, ionosphere and troposphere errors based on the elevation angles.

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves

    Inputs:
      sat_r_es_e (no_sat x 3) ECEF satellite positions (m)
      r_ea_e                ECEF user position (m)
      L_a                   user latitude (rad)
      lambda_a              user longitude (rad)
      GNSS_config
        .no_sat             Number of satellites in constellation
        .mask_angle         Mask angle (deg)
        .SIS_err_SD         Signal in space error SD (m)
        .zenith_iono_err_SD Zenith ionosphere error SD (m)
        .zenith_trop_err_SD Zenith troposphere error SD (m)

    Outputs:
      GNSS_biases (no_sat)  Bias-like GNSS range errors

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    # Constants
    DEG_TO_RAD = 0.01745329252
    RAD_TO_DEG = 1 / DEG_TO_RAD

    # Calculate ECEF to NED coordinate transformation matrix using (2.150)
    cos_lat = math.cos(L_a)
    sin_lat = math.sin(L_a)
    cos_long = math.cos(lambda_a)
    sin_long = math.sin(lambda_a)

    C_e_n = np.zeros((3, 3))
    C_e_n[0, 0] = -sin_lat * cos_long
    C_e_n[0, 1] = -sin_lat * sin_long
    C_e_n[0, 2] = cos_lat
    C_e_n[1, 0] = -sin_long
    C_e_n[1, 1] = cos_long
    C_e_n[2, 0] = -cos_lat * cos_long
    C_e_n[2, 1] = -cos_lat * sin_long
    C_e_n[2, 2] = -sin_lat

    # Loop satellites
    GNSS_biases = np.zeros((GNSS_config.no_sat, 1))
    for j in range(0, GNSS_config.no_sat):
        # Determine ECEF line-of-sight vector using (8.41)
        delta_r = np.transpose(sat_r_es_e[[j], :]) - r_ea_e
        u_as_e = delta_r / math.sqrt(np.matmul(np.transpose(delta_r), delta_r))

        # Convert line-of-sight vector to NED using (8.39) and determine
        # elevation using (8.57)
        elevation = -math.asin(np.matmul(C_e_n[[2], :], u_as_e))

        # Limit the minimum elevation angle to the masking angle
        elevation = max(elevation, GNSS_config.mask_angle * DEG_TO_RAD)

        # Calculate ionosphere and troposphere error SDs using (9.79) and (9.80)
        iono_SD = GNSS_config.zenith_iono_err_SD / math.sqrt(1 - 0.899 * math.cos(elevation) ** 2)
        trop_SD = GNSS_config.zenith_trop_err_SD / math.sqrt(1 - 0.998 * math.cos(elevation) ** 2)

        # Determine range bias
        GNSS_biases[j, 0] = GNSS_config.SIS_err_SD * np.random.randn() + iono_SD * np.random.randn() + trop_SD * np.random.randn()

    return GNSS_biases


def generate_gnss_measurements(time, sat_r_es_e, sat_v_es_e, r_ea_e, L_a, lambda_a, v_ea_e, GNSS_biases, GNSS_config) :
    """
    Generate_GNSS_measurements - Generates a set of pseudo-range and pseudo-
    range rate measurements for all satellites above the elevation mask angle
    and adds satellite positions and velocities to the datesets.

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves

    Inputs:
      time                    Current simulation time
      sat_r_es_e (no_sat x 3) ECEF satellite positions (m)
      sat_v_es_e (no_sat x 3) ECEF satellite velocities (m/s)
      r_ea_e                  ECEF user position (m)
      L_a                     user latitude (rad)
      lambda_a                user longitude (rad)
      v_ea_e                  ECEF user velocity (m/s)
      GNSS_biases (no_sat)    Bias-like GNSS range errors (m)
      GNSS_config
        .no_sat             Number of satellites in constellation
        .mask_angle         Mask angle (deg)
        .code_track_err_SD  Code tracking error SD (m)
        .rate_track_err_SD  Range rate tracking error SD (m/s)
        .rx_clock_offset    Receiver clock offset at time=0 (m)
        .rx_clock_drift     Receiver clock drift at time=0 (m/s)

    Outputs:
      GNSS_measurements     GNSS measurement data:
        Column 1              Pseudo-range measurements (m)
        Column 2              Pseudo-range rate measurements (m/s)
        Columns 3-5           Satellite ECEF position (m)
        Columns 6-8           Satellite ECEF velocity (m/s)
      no_GNSS_meas          Number of satellites for which measurements are
                            supplied

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    # CONSTANTS
    c = 299792458           # Speed of light in m / s
    omega_ie = 7.292115E-5  # Earth rotation rate in rad / s
    DEG_TO_RAD = 0.01745329252

    no_GNSS_meas = 0

    # Calculate ECEF to NED coordinate transformation matrix using (2.150)
    cos_lat = math.cos(L_a)
    sin_lat = math.sin(L_a)
    cos_long = math.cos(lambda_a)
    sin_long = math.sin(lambda_a)

    C_e_n = np.zeros((3, 3))
    C_e_n[0, 0] = -sin_lat * cos_long
    C_e_n[0, 1] = -sin_lat * sin_long
    C_e_n[0, 2] = cos_lat
    C_e_n[1, 0] = -sin_long
    C_e_n[1, 1] = cos_long
    C_e_n[2, 0] = -cos_lat * cos_long
    C_e_n[2, 1] = -cos_lat * sin_long
    C_e_n[2, 2] = -sin_lat

    # Skew symmetric matrix of Earth rate
    Omega_ie = skew_symmetric(np.array([[0], [0], [omega_ie]]))

    # Loop satellites
    GNSS_measurements = np.ndarray((0, 8))

    for j in range(0, GNSS_config.no_sat):
        # Determine ECEF line-of-sight vector using (8.41)
        delta_r = np.transpose(sat_r_es_e[[j], :]) - r_ea_e
        approx_range = math.sqrt(np.matmul(np.transpose(delta_r), delta_r))
        u_as_e = delta_r / approx_range

        # Convert line-of-sight vector to NED using (8.39) and determine
        # elevation using (8.57)
        elevation = -math.asin(np.matmul(C_e_n[[2], :], u_as_e))

        # Determine if satellite is above the masking angle
        if (elevation >= GNSS_config.mask_angle * DEG_TO_RAD):
            # Increment number of measurements
            no_GNSS_meas = no_GNSS_meas + 1

            # Calculate frame rotation during signal transit time using (8.36)
            C_e_I = np.zeros((3, 3))
            C_e_I[0, 0] = 1
            C_e_I[0, 1] = omega_ie * approx_range / c
            C_e_I[1, 0] = -omega_ie * approx_range / c
            C_e_I[1, 1] = 1
            C_e_I[2, 2] = 1

            # Calculate range using (8.35)
            delta_r = np.matmul(C_e_I, np.transpose(sat_r_es_e[[j], :])) - r_ea_e
            range_ = math.sqrt(np.matmul(np.transpose(delta_r), delta_r))

            # Calculate range rate using (8.44)
            temp = np.matmul(C_e_I, np.transpose(sat_v_es_e[[j], :]) + np.matmul(Omega_ie, np.transpose(sat_r_es_e[[j], :]))) - (v_ea_e + np.matmul(Omega_ie, r_ea_e))
            range_rate = np.matmul(np.transpose(u_as_e), temp)

            # Calculate pseudo-range measurement
            measurements = np.zeros((1, 8))
            measurements[0, 0] = range_ + GNSS_biases[j, 0] + GNSS_config.rx_clock_offset + GNSS_config.rx_clock_drift * time + GNSS_config.code_track_err_SD * np.random.randn()

            # Calculate pseudo-range rate measurement
            measurements[0, 1] = range_rate + GNSS_config.rx_clock_drift + GNSS_config.rate_track_err_SD * np.random.randn()

            # Append satellite position and velocity to output data
            measurements[[0], 2: 5] = sat_r_es_e[[j], :]
            measurements[[0], 5:] = sat_v_es_e[[j], :]

            GNSS_measurements = np.vstack((GNSS_measurements, measurements))

    return GNSS_measurements, no_GNSS_meas
