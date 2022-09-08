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
from read_profile import *
from Simulation.engine.integrated_navigation import *
from Simulation.plot.plot import *
from write_profile_error import *

# CONSTANTS
DEG_TO_RAD = 0.01745329252
RAD_TO_DEG = 1 / DEG_TO_RAD
MICRO_G_TO_METERS_PER_SECOND_SQUARED = 9.80665 * 1e-6

# CONFIGURATION
input_profile_name = "/Users/junc/workspace/projects/Nav/Data/profiles/Profile_1.csv"                     # Input truth motion profile filename
output_profile_name = "/Users/junc/workspace/projects/Nav/Data/profiles/INS_GNSS_Demo_1_Profile.csv"      # Output motion profile and error filenames
output_error_name = "/Users/junc/workspace/projects/Nav/Data/profiles/INS_GNSS_Demo_1_Errors.csv"


class InitializationError:
    def __init__(self):
        self.delta_r_eb_n = np.array([[4], [2], [3]])                       # Position initialization error (m; N, E, D)
        self.delta_v_eb_n = np.array([[0.05], [-0.05], [0.1]])              # Velocity initialization error (m/s; N, E, D)
        self.delta_eul_nb_n = np.array([[-0.05], [0.04], [1]]) * DEG_TO_RAD # Attitude initialization error (rad; N, E, D)


class IMUErrors:
    def __init__(self):
        self.b_a = np.array([[900], [-1300], [800]]) * MICRO_G_TO_METERS_PER_SECOND_SQUARED     # ACC Biases (micro-g, converted to m/s^2; body axes)
        self.b_g = np.array([[-9], [13], [-8]]) * DEG_TO_RAD / 3600                             # GYR biases (deg/hour, converted to rad/sec; body axes)
        self.M_a = np.array([[500, -300, 200],                                                  # ACC scale factor and cross coupling errors
                             [-150, -600, 250],                                                 # (ppm, converted to unitless, body axes)
                             [-250, 100, 450]]) * 1e-6
        self.M_g = np.array([[400, -300, 250],                                                  # GYR scale factor and cross coupling errors
                             [0, -300, -150],                                                   # (ppm, converted to unitless, body axes)
                             [0, 0, -350]]) * 1e-6
        self.G_g = np.array([[0.9, -1.1, -0.6],                                                 # GYR g dependent biases
                             [-0.5, 1.9, -1.6],                                                 # (deg/hour/g, converted to rad-sec/m, body axes)
                             [0.3, 1.1, -1.3]]) * DEG_TO_RAD / (3600 * 9.80665)
        self.accel_noise_root_PSD = 100 * MICRO_G_TO_METERS_PER_SECOND_SQUARED                  # ACC Noise root PSD (micro-g/Hz, converted to m s^-1.5)
        self.gyro_noise_root_PSD = 0.01 * DEG_TO_RAD / 60                                       # GYR Noise root PSD (deg per root hour, converted to rad s^-0.5)
        self.accel_quant_level = 1e-2                                                           # Acc quantization level (m/s^2)
        self.gyro_quant_level = 2e-4                                                            # GYR quantization level (rad/s)


class GNSSConfig:
    def __init__(self):
        self.epoch_interval = 0.5                           # Interval between GNSS epochs (s)
        self.init_est_r_ea_e = np.array([[0], [0], [0]])    # Initial estimated position (m; ECEF)
        self.no_sat = 30                                    # Number of satellites in constellation
        self.r_os = 2.656175E7                              # Orbital radius of satellites (m)
        self.inclination = 55                               # Inclination angle of satellites (deg)
        self.const_delta_lambda = 0                         # Longitude offset of constellation (deg)
        self.const_delta_t = 0                              # Timing offset of constellation (s)
        self.mask_angle = 10                                # Mask angle (deg)
        self.SIS_err_SD = 1                                 # Signal in space error SD (m) *Give residual where corrections are applied
        self.zenith_iono_err_SD = 2                         # Zenith ionosphere error SD (m) *Give residual where corrections are applied
        self.zenith_trop_err_SD = 0.2                       # Zenith troposphere error SD (m) *Give residual where corrections are applied
        self.code_track_err_SD = 1                          # Code tracking error SD (m) *Can extend to account for multipath
        self.rate_track_err_SD = 0.02                       # Range rate tracking error SD (m/s) *Can extend to account for multipath
        self.rx_clock_offset = 10000                        # Receiver clock offset at time=0 (m);
        self.rx_clock_drift = 100                           # Receiver clock drift at time=0 (m/s);


class TCKFConfig:
    def __init__(self):
        self.init_att_unc = 1 * DEG_TO_RAD                                          # Initial attitude uncertainty per axis (deg, converted to rad)
        self.init_vel_unc = 0.1                                                     # Initial velocity uncertainty per axis (m/s)
        self.init_pos_unc = 10                                                      # Initial position uncertainty per axis (m)
        self.init_b_a_unc = 1000 * MICRO_G_TO_METERS_PER_SECOND_SQUARED             # Initial accelerometer bias uncertainty per instrument (micro-g, converted to m/s^2)
        self.init_b_g_unc = 10 * DEG_TO_RAD / 3600                                  # Initial gyro bias uncertainty per instrument (deg/hour, converted to rad/sec)
        self.init_clock_offset_unc = 10                                             # Initial clock offset uncertainty per axis (m)
        self.init_clock_drift_unc = 0.1                                             # Initial clock drift uncertainty per axis (m/s)
        self.gyro_noise_PSD = (0.02 * DEG_TO_RAD / 60) ** 2                         # Gyro noise PSD (deg^2 per hour, converted to rad^2/s)
        self.accel_noise_PSD = (200 * MICRO_G_TO_METERS_PER_SECOND_SQUARED) ** 2    # Accelerometer noise PSD (micro-g^2 per Hz, converted to m^2 s^-3)
        self.accel_bias_PSD = 1.0E-7                                                # Accelerometer bias random walk PSD (m^2 s^-5)
        self.gyro_bias_PSD = 2.0E-12                                                # Gyro bias random walk PSD (rad^2 s^-3)
        self.clock_freq_PSD = 1                                                     # Receiver clock frequency-drift PSD (m^2/s^3)
        self.clock_phase_PSD = 1                                                    # Receiver clock phase-drift PSD (m^2/s)
        self.pseudo_range_SD = 2.5                                                  # Pseudo-range measurement noise SD (m)
        self.range_rate_SD = 0.1                                                    # Pseudo-range rate measurement noise SD (m/s)


rand_stream = np.random.seed(1)         # Seeding of the random number generator for reproducibility. Change
                                        # this value for a different random number sequence (may not work in Octave).


if __name__ == '__main__':
    """
    INS_GNSS_Demo_1
    SCRIPT Tightly coupled INS/GNSS demo:
      Profile_1 (60s artificial car motion with two 90 deg turns)
      Tactical-grade IMU

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    Created 12/4/12 by Paul Groves

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    print('INS GNSS Demo 1 Profile 1')

    # Input truth motion profile from .csv format file
    in_profile = read_profile(input_profile_name)
    no_epochs = in_profile.shape[0]

    # Tightly coupled ECEF Inertial navigation and GNSS integrated navigation
    # simulation
    initialization_error = InitializationError()
    imu_errors = IMUErrors()
    gnss_config = GNSSConfig()
    tckf_config = TCKFConfig()
    out_profile, out_errors, out_IMU_bias_est, out_clock, out_KF_SD = tightly_coupled_ins_gnss(in_profile, no_epochs, initialization_error, imu_errors, gnss_config, tckf_config)

    # Plot the input motion profile and the errors
    plot_profile(in_profile)
    plot_errors(out_errors)

    # Write output profile and errors file
    write_profile(output_profile_name, out_profile)
    write_errors(output_error_name, out_errors)
