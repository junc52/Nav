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

from Simulation.utils.curvilinear_conversion import *
from Simulation.utils.frame_transform import ctm_to_euler


def calculate_errors_ned(est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n,
                         true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n):
    """
    Calculate_errors_NED - Calculates the position, velocity, and attitude
    errors of a NED navigation solution.

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 3/4/2012 by Paul Groves

    Inputs:
      est_L_b       latitude solution (rad)
      est_lambda_b  longitude solution (rad)
      est_h_b       height solution (m)
      est_v_eb_n    velocity solution of body frame w.r.t. ECEF frame,
                    resolved along north, east, and down (m/s)
      est_C_b_n     body-to-NED coordinate transformation matrix solution
      true_L_b      true latitude (rad)
      true_lambda_b true longitude (rad)
      true_h_b      true height (m)
      true_v_eb_n   true velocity of body frame w.r.t. ECEF frame, resolved
                    along north, east, and down (m/s)
      C_b_n         true body-to-NED coordinate transformation matrix

    Outputs:
      delta_r_eb_n     position error resolved along NED (m)
      delta_v_eb_n     velocity error resolved along NED (m/s)
      delta_eul_nb_n   attitude error as NED Euler angles (rad)
                       These are expressed about north, east, and down
    """
    # Position error calculation, using (2.119)
    R_N, R_E = radii_of_curvature(true_L_b)
    delta_r_eb_n = np.zeros((3, 1))
    delta_r_eb_n[0, 0] = (est_L_b - true_L_b) * (R_N + true_h_b)
    delta_r_eb_n[1, 0] = (est_lambda_b - true_lambda_b) * (R_E + true_h_b) * math.cos(true_L_b)
    delta_r_eb_n[2, 0] = -(est_h_b - true_h_b)

    # Velocity error calculation
    delta_v_eb_n = est_v_eb_n - true_v_eb_n

    # Attitude error calculation, using (5.109) and (5.111)
    delta_C_b_n = np.matmul(est_C_b_n, np.transpose(true_C_b_n))
    delta_eul_nb_n = -ctm_to_euler(delta_C_b_n)

    return delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n
