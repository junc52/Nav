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
from Simulation.utils.gravitation_gravity_model import *
from Simulation.utils.frame_transform import skew_symmetric
from Simulation.utils.curvilinear_conversion import radii_of_curvature


def kinematics_eci(tor_i, C_b_i, old_C_b_i, v_ib_i, old_v_ib_i, r_ib_i):
    """
    Kinematics_ECI - calculates specific force and angular rate from input
    w.r.t and resolved along ECI-frame axes

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 1/4/2012 by Paul Groves

    Inputs:
      tor_i         time interval between epochs (s)
      C_b_i         body-to-ECI-frame coordinate transformation matrix
      old_C_b_i     previous body-to-ECI-frame coordinate transformation matrix
      v_ib_i        velocity of body frame w.r.t. ECI frame, resolved along
                    ECI-frame axes (m/s)
      old_v_ib_i    previous velocity of body frame w.r.t. ECI frame, resolved
                    along ECI-frame axes (m/s)
      r_ib_i        Cartesian position of body frame w.r.t. ECI frame, resolved
                    along ECI-frame axes (m)
    Outputs:
      f_ib_b        specific force of body frame w.r.t. ECI frame, resolved
                    along body-frame axes, averaged over time interval (m/s^2)
      omega_ib_b    angular rate of body frame w.r.t. ECI frame, resolved
                    about body-frame axes, averaged over time interval (rad/s)
    """
    if tor_i > 0:
        # Obtain coordinate transformation matrix from the old attitude to the new using (5.68), (5.109)
        C_old_new = np.matmul(np.transpose(C_b_i), old_C_b_i)               # attitude update

        # Calculate the approximate angular rate using (5.74)
        alpha_ib_b = np.zeros((3, 1))                                       # attitude increment
        alpha_ib_b[0, 0] = 0.5 * (C_old_new[1, 2] - C_old_new[2, 1])
        alpha_ib_b[1, 0] = 0.5 * (C_old_new[2, 0] - C_old_new[0, 2])
        alpha_ib_b[2, 0] = 0.5 * (C_old_new[0, 1] - C_old_new[1, 0])

        # Calculate and apply the scaling factor (5.74)
        temp = math.acos(0.5 * (C_old_new[0, 0] + C_old_new[1, 1] + C_old_new[2, 2] - 1.0))

        if temp > 2e-5:          # scaling is 1 is temp is less than this
            alpha_ib_b = alpha_ib_b * (temp/math.sin(temp))

        # Calculate the angular rate using section 5.2.1
        omega_ib_b = alpha_ib_b / tor_i

        # Calculate the specific force resolved about ECI-frame axes from (5.18), (5.19), (5.20)
        f_ib_i = ((v_ib_i - old_v_ib_i) / tor_i) - gravitation_eci(r_ib_i)

        # Calculate the average body-to-ECI-frame coordinate transformation
        # matrix over the update interval using (5.84)
        mag_alpha = np.linalg.norm(alpha_ib_b)
        Alpha_ib_b = skew_symmetric(alpha_ib_b)

        if mag_alpha > 1.e-8:
            ave_C_b_i = np.matmul(old_C_b_i, (np.identity(3) + ((1 - math.cos(mag_alpha)) / mag_alpha ** 2)
                                              * Alpha_ib_b + ((1 - math.sin(mag_alpha) / mag_alpha) / mag_alpha ** 2)
                                              * np.matmul(Alpha_ib_b, Alpha_ib_b)))

        else:
            ave_C_b_i = old_C_b_i

        # Transform specific force to body-frame resolving axes using (5.81)
        f_ib_b = np.matmul(np.linalg.inv(ave_C_b_i), f_ib_i)

    else:
        # If time interval is zero, set angular rate and specific force to zero
        omega_ib_b = np.zeros((3, 1))
        f_ib_b = np.zeros((3, 1))

    return f_ib_b, omega_ib_b


def kinematics_ecef(tor_i, C_b_e, old_C_b_e, v_eb_e, old_v_eb_e, r_eb_e):
    """
    Kinematics_ECEF - calculates specific force and angular rate from input
    w.r.t and resolved along ECEF-frame axes

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 1/4/2012 by Paul Groves

    Inputs:
      tor_i         time interval between epochs (s)
      C_b_e         body-to-ECEF-frame coordinate transformation matrix
      old_C_b_e     previous body-to-ECEF-frame coordinate transformation matrix
      v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
                    ECEF-frame axes (m/s)
      old_v_eb_e    previous velocity of body frame w.r.t. ECEF frame, resolved
                    along ECEF-frame axes (m/s)
      r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
                    along ECEF-frame axes (m)
    Outputs:
      f_ib_b        specific force of body frame w.r.t. ECEF frame, resolved
                    along body-frame axes, averaged over time interval (m/s^2)
      omega_ib_b    angular rate of body frame w.r.t. ECEF frame, resolved
                    about body-frame axes, averaged over time interval (rad/s)
    """
    # CONSTANT
    omega_ie = 7.292115e-5  #  Earth rotation rate(rad / s)

    if tor_i > 0:
        # From (2.145) determine the Earth rotation over the update interval
        # C_Earth = C_e_i' * old_C_e_i
        alpha_ie = omega_ie * tor_i
        C_Earth = np.zeros((3, 3))
        C_Earth[0, 0] = math.cos(alpha_ie)
        C_Earth[0, 1] = math.sin(alpha_ie)
        C_Earth[1, 0] = -math.sin(alpha_ie)
        C_Earth[1, 1] = math.cos(alpha_ie)
        C_Earth[2, 2] = 1

        # Obtain coordinate transformation matrix from the old attitude (w.r.t.
        # an inertial frame) to the new
        C_old_new = np.matmul(np.matmul(np.transpose(C_b_e), C_Earth), old_C_b_e)            # attitude update

        # Calculate the approximate angular rate w.r.t. an inertial frame using (5.74)
        alpha_ib_b = np.zeros((3, 1))                                       # attitude increment
        alpha_ib_b[0, 0] = 0.5 * (C_old_new[1, 2] - C_old_new[2, 1])
        alpha_ib_b[1, 0] = 0.5 * (C_old_new[2, 0] - C_old_new[0, 2])
        alpha_ib_b[2, 0] = 0.5 * (C_old_new[0, 1] - C_old_new[1, 0])

        # Calculate and apply the scaling factor (5.74)
        temp = math.acos(0.5 * (C_old_new[0, 0] + C_old_new[1, 1] + C_old_new[2, 2] - 1.0))

        if temp > 2e-5:          # scaling is 1 is temp is less than this
            alpha_ib_b = alpha_ib_b * (temp/math.sin(temp))

        # Calculate the angular rate using section 5.2.1
        omega_ib_b = alpha_ib_b / tor_i

        # Calculate the specific force resolved about ECEF-frame axes from (5.36)
        f_ib_e = ((v_eb_e - old_v_eb_e) / tor_i) - gravity_ecef(r_eb_e) + 2 * np.matmul(skew_symmetric(np.array([[0], [0], [omega_ie]])), old_v_eb_e)

        # Calculate the average body-to-ECI-frame coordinate transformation
        # matrix over the update interval using (5.84) and (5.85)
        mag_alpha = np.linalg.norm(alpha_ib_b)
        Alpha_ib_b = skew_symmetric(alpha_ib_b)

        if mag_alpha > 1.e-8:
            ave_C_b_e = np.matmul(old_C_b_e, (np.identity(3) + ((1 - math.cos(mag_alpha)) / mag_alpha ** 2)
                                     * Alpha_ib_b + ((1 - math.sin(mag_alpha) / mag_alpha) / mag_alpha ** 2)
                                     * np.matmul(Alpha_ib_b, Alpha_ib_b)) - 0.5 * np.matmul(skew_symmetric(np.array([[0], [0], [alpha_ie]])), old_C_b_e))

        else:
            ave_C_b_e = old_C_b_e - 0.5 * np.matmul(skew_symmetric(np.array([[0], [0], [alpha_ie]])), old_C_b_e)

        # Transform specific force to body-frame resolving axes using (5.81)
        # TODO check the equation
        f_ib_b = np.matmul(np.linalg.inv(ave_C_b_e), f_ib_e)

    else:
        # If time interval is zero, set angular rate and specific force to zero
        omega_ib_b = np.zeros((3, 1))
        f_ib_b = np.zeros((3, 1))

    return f_ib_b, omega_ib_b


def kinematics_ned(tor_i, C_b_n, old_C_b_n, v_eb_n, old_v_eb_n, L_b, h_b, old_L_b, old_h_b):
    """
    inematics_NED - calculates specific force and angular rate from input
    w.r.t and resolved along north, east, and down

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 1/4/2012 by Paul Groves

    Inputs:
      tor_i         time interval between epochs (s)
      C_b_n         body-to-NED coordinate transformation matrix
      old_C_b_n     previous body-to-NED coordinate transformation matrix
      v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
                    north, east, and down (m/s)
      old_v_eb_n    previous velocity of body frame w.r.t. ECEF frame, resolved
                    along north, east, and down (m/s)
      L_b           latitude (rad)
      h_b           height (m)
      old_L_b       previous latitude (rad)
      old_h_b       previous height (m)
    Outputs:
      f_ib_b        specific force of body frame w.r.t. ECEF frame, resolved
                    along body-frame axes, averaged over time interval (m/s^2)
      omega_ib_b    angular rate of body frame w.r.t. ECEF frame, resolved
                    about body-frame axes, averaged over time interval (rad/s)
    """

    # CONSTANT
    omega_ie = 7.292115e-5      # Earth rotation rate (rad/s)

    if tor_i > 0:
        # From (2.123) , determine the angular rate of the ECEF frame
        # w.r.t the ECI frame, resolved about NED
        omega_ie_n = omega_ie * np.array([[math.cos(old_L_b)], [0], [-math.sin(old_L_b)]])

        # From (5.44), determine the angular rate of the NED frame
        # w.r.t the ECEF frame, resolved about NED
        old_R_N, old_R_E = radii_of_curvature(old_L_b)
        R_N, R_E = radii_of_curvature(L_b)
        old_omega_en_n = np.array([[old_v_eb_n[1, 0] / (old_R_E + old_h_b)], [-old_v_eb_n[0, 0] / (old_R_N + old_h_b)], [-old_v_eb_n[1, 0] * math.tan(old_L_b) / (old_R_E + old_h_b)]])
        omega_en_n = np.array([[v_eb_n[1, 0] / (R_E + h_b)], [-v_eb_n[0, 0] / (R_N + h_b)], [-v_eb_n[1, 0] * math.tan(L_b) / (R_E + h_b)]])

        # Obtain coordinate transformation matrix from the old attitude (w.r.t.
        # an inertial frame) to the new using (5.77)
        C_old_new = np.matmul(np.matmul(np.transpose(C_b_n), np.eye(3) + skew_symmetric(omega_ie_n + 0.5 * omega_en_n + 0.5 * old_omega_en_n) * tor_i), old_C_b_n)

        # Calculate the approximate angular rate w.r.t. an inertial frame
        alpha_ib_b = np.zeros((3, 1))  # attitude increment
        alpha_ib_b[0, 0] = 0.5 * (C_old_new[1, 2] - C_old_new[2, 1])
        alpha_ib_b[1, 0] = 0.5 * (C_old_new[2, 0] - C_old_new[0, 2])
        alpha_ib_b[2, 0] = 0.5 * (C_old_new[0, 1] - C_old_new[1, 0])

        # Calculate and apply the scaling factor
        temp = math.acos(0.5 * (C_old_new[0, 0] + C_old_new[1, 1] + C_old_new[2, 2] - 1.0))

        if temp > 2e-5:  # scaling is 1 is temp is less than this
            alpha_ib_b = alpha_ib_b * (temp / math.sin(temp))

        # Calculate the angular rate using section 5.2.1
        omega_ib_b = alpha_ib_b / tor_i

        # Calculate the specific force resolved about ECEF-frame axes From (5.54)
        f_ib_n = ((v_eb_n - old_v_eb_n) / tor_i) - gravity_ned(old_L_b, old_h_b) + np.matmul(skew_symmetric(old_omega_en_n + 2 * omega_ie_n), old_v_eb_n)

        # Calculate the average body-to-NED coordinate transformation
        # matrix over the update interval using (5.84) and (5.86)
        mag_alpha = np.linalg.norm(alpha_ib_b)
        Alpha_ib_b = skew_symmetric(alpha_ib_b)

        if mag_alpha > 1.e-8:
            ave_C_b_n = np.matmul(old_C_b_n, (np.identity(3) + ((1 - math.cos(mag_alpha)) / mag_alpha ** 2)
                                     * Alpha_ib_b + ((1 - math.sin(mag_alpha) / mag_alpha) / mag_alpha ** 2)
                                     * np.matmul(Alpha_ib_b, Alpha_ib_b)) - 0.5 * np.matmul(skew_symmetric(old_omega_en_n + omega_ie_n), old_C_b_n))

        else:
            ave_C_b_n = old_C_b_n - 0.5 * np.matmul(skew_symmetric(old_omega_en_n + omega_ie_n), old_C_b_n)

        # Transform specific force to body-frame resolving axes using (5.81)
        f_ib_b = np.matmul(np.linalg.inv(ave_C_b_n), f_ib_n)
    else:
        # If time interval is zero, set angular rate and specific force to zero
        omega_ib_b = np.zeros((3, 1))
        f_ib_b = np.zeros((3, 1))

    return f_ib_b, omega_ib_b