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
from Simulation.utils.frame_transform import *
# from utils.curvilinear_conversion import *


def initialize_gnss_kf(est_r_ea_e, est_v_ea_e, est_clock, GNSS_KF_config):
    """
    nitialize_GNSS_KF - Initializes the GNSS EKF state estimates and error
    covariance matrix

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves

    Inputs:
      est_r_ea_e            estimated ECEF user position (m)
      est_v_ea_e            estimated ECEF user velocity (m/s)
      est_clock             estimated receiver clock offset (m) and drift (m/s)
      GNSS_KF_config
        .init_pos_unc           Initial position uncertainty per axis (m)
        .init_vel_unc           Initial velocity uncertainty per axis (m/s)
        .init_clock_offset_unc  Initial clock offset uncertainty per axis (m)
        .init_clock_drift_unc   Initial clock drift uncertainty per axis (m/s)

    Outputs:
      x_est                 Kalman filter estimates:
        Rows 1-3            estimated ECEF user position (m)
        Rows 4-6            estimated ECEF user velocity (m/s)
        Row 7               estimated receiver clock offset (m)
        Row 8               estimated receiver clock drift (m/s)
      P_matrix              state estimation error covariance matrix

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    # Initialize state estimates
    x_est = np.zeros((8, 1))
    x_est[0:3, [0]] = est_r_ea_e
    x_est[3:6, [0]] = est_v_ea_e
    x_est[6:, [0]]  = est_clock

    # Initialize error covariance matrix
    P_matrix = np.zeros((8, 8))
    P_matrix[0, 0] = GNSS_KF_config.init_pos_unc ** 2
    P_matrix[1, 1] = GNSS_KF_config.init_pos_unc ** 2
    P_matrix[2, 2] = GNSS_KF_config.init_pos_unc ** 2
    P_matrix[3, 3] = GNSS_KF_config.init_vel_unc ** 2
    P_matrix[4, 4] = GNSS_KF_config.init_vel_unc ** 2
    P_matrix[5, 5] = GNSS_KF_config.init_vel_unc ** 2
    P_matrix[6, 6] = GNSS_KF_config.init_clock_offset_unc ** 2
    P_matrix[7, 7] = GNSS_KF_config.init_clock_drift_unc ** 2

    return x_est, P_matrix


def initialize_tc_P_matrix(TC_KF_config):
    """
    Initialize_TC_P_matrix - Initializes the tightly coupled INS/GNSS EKF
    error covariance matrix

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 12/4/2012 by Paul Groves

    Inputs:
      TC_KF_config
        .init_att_unc           Initial attitude uncertainty per axis (rad)
        .init_vel_unc           Initial velocity uncertainty per axis (m/s)
        .init_pos_unc           Initial position uncertainty per axis (m)
        .init_b_a_unc           Initial accel. bias uncertainty (m/s^2)
        .init_b_g_unc           Initial gyro. bias uncertainty (rad/s)
        .init_clock_offset_unc  Initial clock offset uncertainty per axis (m)
        .init_clock_drift_unc   Initial clock drift uncertainty per axis (m/s)

    Outputs:
      P_matrix              state estimation error covariance matrix

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    # Initialize error covariance matrix
    P_matrix = np.zeros((17, 17))
    P_matrix[0: 3, 0: 3] = np.identity(3) * TC_KF_config.init_att_unc ** 2
    P_matrix[3: 6, 3: 6] = np.identity(3) * TC_KF_config.init_vel_unc ** 2
    P_matrix[6: 9, 6: 9] = np.identity(3) * TC_KF_config.init_pos_unc ** 2
    P_matrix[9: 12, 9: 12] = np.identity(3) * TC_KF_config.init_b_a_unc ** 2
    P_matrix[12: 15, 12: 15] = np.identity(3) * TC_KF_config.init_b_g_unc ** 2
    P_matrix[15, 15] = TC_KF_config.init_clock_offset_unc ** 2
    P_matrix[16, 16] = TC_KF_config.init_clock_drift_unc ** 2

    return P_matrix


def initialize_lc_P_matrix(LC_KF_config):
    """
    Initialize_LC_P_matrix - Initializes the loosely coupled INS/GNSS KF
    error covariance matrix

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 12/4/2012 by Paul Groves

    Inputs:
      TC_KF_config
        .init_att_unc           Initial attitude uncertainty per axis (rad)
        .init_vel_unc           Initial velocity uncertainty per axis (m/s)
        .init_pos_unc           Initial position uncertainty per axis (m)
        .init_b_a_unc           Initial accel. bias uncertainty (m/s^2)
        .init_b_g_unc           Initial gyro. bias uncertainty (rad/s)

    Outputs:
      P_matrix              state estimation error covariance matrix

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    P_matrix = np.zeros((15,15))
    P_matrix[0: 3, 0: 3] = np.identity(3) * LC_KF_config.init_att_unc ** 2
    P_matrix[3: 6, 3: 6] = np.identity(3) * LC_KF_config.init_vel_unc ** 2
    P_matrix[6: 9, 6: 9] = np.identity(3) * LC_KF_config.init_pos_unc ** 2
    P_matrix[9: 12, 9: 12] = np.identity(3) * LC_KF_config.init_b_a_unc ** 2
    P_matrix[12: 15, 12: 15] = np.identity(3) * LC_KF_config.init_b_g_unc ** 2

    return P_matrix
