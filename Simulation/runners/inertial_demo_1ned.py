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
from Simulation.engine.inertial_navigation import *
from Simulation.plot.plot import *
from write_profile_error import *

# CONSTANTS
DEG_TO_RAD = 0.01745329252
RAD_TO_DEG = 1 / DEG_TO_RAD
MICRO_G_TO_METERS_PER_SECOND_SQUARED = 9.80665 * 1e-6

# CONFIGURATION
input_profile_name = "/Users/junc/workspace/projects/Nav/Data/profiles/Profile_1.csv"                    # Input truth motion profile filename
output_profile_name = "/Users/junc/workspace/projects/Nav/Data/profiles/Inertial_Demo_1NED_Profile.csv"  # Output motion profile and error filenames
output_error_name = "/Users/junc/workspace/projects/Nav/Data/profiles/Inertial_Demo_1NED_Errors.csv"


class InitializationError:
    def __init__(self):
        self.delta_r_eb_n = np.array([[4], [2], [3]])                       # Position initialization error (m; N, E, D)
        self.delta_v_eb_n = np.array([[0.05], [-0.05], [0.1]])              # Velocity initialization error (m/s; N, E, D)
        self.delta_eul_nb_n = np.array([[-0.05], [0.04], [1]]) * DEG_TO_RAD # Attitude initialization error (rad; N, E, D)


class IMUErrors:
    def __init__(self):
        self.b_a = np.array([[900], [-1300], [800]])  * MICRO_G_TO_METERS_PER_SECOND_SQUARED    # ACC Biases (m/s^2, body axes)
        self.b_g = np.array([[-9], [13], [-8]]) *DEG_TO_RAD / 3600                              # GYR biases (rad/s, body axes)
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


rand_stream = np.random.seed(1)         # Seeding of the random number generator for reproducibility. Change
                                        # this value for a different random number sequence (may not work in Octave).


if __name__ == '__main__':
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
    """
    print('Inertial Demo NED 1 Profile 1')

    # Input truth motion profile from .csv format file
    in_profile = read_profile(input_profile_name)
    no_epochs = in_profile.shape[0]

    # NED Inertial navigation simulation
    initialization_error = InitializationError()
    imu_errors = IMUErrors()
    out_profile, out_errors = inertial_navigation_ned(in_profile, no_epochs, initialization_error, imu_errors)

    # Plot the input motion profile and the errors
    plot_profile(in_profile)
    plot_errors(out_errors)

    # Write output profile and errors file
    write_profile(output_profile_name, out_profile)
    write_errors(output_error_name, out_errors)
