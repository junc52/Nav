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
from Simulation.engine.gnss_navigation import *
from Simulation.plot.plot import *
from write_profile_error import *

# CONSTANTS
DEG_TO_RAD = 0.01745329252
RAD_TO_DEG = 1 / DEG_TO_RAD
MICRO_G_TO_METERS_PER_SECOND_SQUARED = 9.80665 * 1e-6

# CONFIGURATION
input_profile_name = "/Users/junc/workspace/projects/Nav/Data/profiles/Profile_1.csv"             # Input truth motion profile filename
output_profile_name = "/Users/junc/workspace/projects/Nav/Data/profiles/GNSS_Demo_2_Profile.csv"  # Output motion profile and error filenames
output_error_name = "/Users/junc/workspace/projects/Nav/Data/profiles/GNSS_Demo_2_Errors.csv"


class GNSSConfig:
    def __init__(self):
        self.epoch_interval = 1                             # Interval between GNSS epochs (s)
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

class GNSSKFConfig:
    def __init__(self):
        self.init_pos_unc = 10                              # Initial position uncertainty per axis (m)
        self.init_vel_unc = 0.1                             # Initial velocity uncertainty per axis (m/s)
        self.init_clock_offset_unc = 10                     # Initial clock offset uncertainty per axis (m)
        self.init_clock_drift_unc = 0.1                     # Initial clock drift uncertainty per axis (m/s)
        self.accel_PSD = 10                                 # Acceleration PSD per axis (m^2/s^3)
        self.clock_freq_PSD = 1                             # Receiver clock frequency-drift PSD (m^2/s^3)
        self.clock_phase_PSD = 1                            # Receiver clock phase-drift PSD (m^2/s)
        self.pseudo_range_SD = 2.5                          # Pseudo-range measurement noise SD (m)
        self.range_rate_SD = 0.05                           # Pseudo-range rate measurement noise SD (m/s)

rand_stream = np.random.seed(1)         # Seeding of the random number generator for reproducibility. Change
                                        # this value for a different random number sequence (may not work in Octave).


if __name__ == '__main__':
    """GNSS_Demo_2
    SCRIPT Stand-alone GNSS demo with kalman filter solution:
    Profile_1 (60s artificial car motion with two 90 deg turns)

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    Created 12/4/12 by Paul Groves

    Copyright 2012, Paul Groves
    """

    print('GNSS Demo 2 Profile 1')

    # Input truth motion profile from .csv format file
    in_profile = read_profile(input_profile_name)
    no_epochs = in_profile.shape[0]
    # NED Inertial navigation simulation
    gnss_config = GNSSConfig()
    gnss_kf_config = GNSSKFConfig()
    out_profile, out_errors, out_clock, out_KF_SD = gnss_kalman_filter(in_profile, no_epochs, gnss_config, gnss_kf_config)

    # Plot the input motion profile and the errors
    plot_profile(in_profile)
    plot_errors(out_errors)

    # Write output profile and errors file
    write_profile(output_profile_name, out_profile)
    write_errors(output_error_name, out_errors)