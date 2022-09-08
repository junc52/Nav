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

from Simulation.utils.frame_transform import skew_symmetric
from Simulation.utils.gravitation_gravity_model import *
from Simulation.utils.curvilinear_conversion import radii_of_curvature


def nav_equations_eci(tor_i, old_r_ib_i, old_v_ib_i, old_C_b_i, f_ib_b, omega_ib_b):
    """
    Nav_equations_ECI - Runs precision ECI-frame inertial navigation
    equations

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 1/4/2012 by Paul Groves

    Inputs:
      tor_i         time interval between epochs (s)
      old_r_ib_i    previous Cartesian position of body frame w.r.t. ECI
                    frame, resolved along ECI-frame axes (m)
      old_C_b_i     previous body-to-ECI-frame coordinate transformation matrix
      old_v_ib_i    previous velocity of body frame w.r.t. ECI frame, resolved
                    along ECI-frame axes (m/s)
      f_ib_b        specific force of body frame w.r.t. ECI frame, resolved
                    along body-frame axes, averaged over time interval (m/s^2)
      omega_ib_b    angular rate of body frame w.r.t. ECI frame, resolved
                    about body-frame axes, averaged over time interval (rad/s)
    Outputs:
      r_ib_i        Cartesian position of body frame w.r.t. ECI frame, resolved
                    along ECI-frame axes (m)
      v_ib_i        velocity of body frame w.r.t. ECI frame, resolved along
                    ECI-frame axes (m/s)
      C_b_i         body-to-ECI-frame coordinate transformation matrix
    """
    # ATTITUDE UPDATE
    # Calculate attitude increment, magnitude, and skew-symmetric matrix
    alpha_ib_b = omega_ib_b * tor_i
    mag_alpha = math.sqrt(np.matmul(np.transpose(alpha_ib_b), alpha_ib_b))
    Alpha_ib_b = skew_symmetric(alpha_ib_b)

    # Obtain coordinate transformation matrix from the new attitude to the old
    # using Rodrigues' formula, (5.73)
    if mag_alpha > 1e-8:
        C_new_old = np.eye(3) + (math.sin(mag_alpha) / mag_alpha) * Alpha_ib_b + \
                    np.matmul(((1 - math.cos(mag_alpha)) / mag_alpha ** 2) * Alpha_ib_b, Alpha_ib_b)
    else :
        C_new_old = np.eye(3) + Alpha_ib_b

    # Update attitude
    C_b_i = np.matmul(old_C_b_i, C_new_old)

    # SPECIFIC FORCE FRAME TRANSFORMATION
    # Calculate the average body-to-ECI-frame coordinate transformation
    # matrix over the update interval using (5.84)
    if mag_alpha > 1e-8:
        first_order = ((1 - math.cos(mag_alpha)) / mag_alpha ** 2) * Alpha_ib_b
        second_order = ((1 - (math.sin(mag_alpha) / mag_alpha)) / mag_alpha ** 2) * np.matmul(Alpha_ib_b, Alpha_ib_b)
        C_new_old = np.identity(3) + first_order + second_order
        ave_C_b_i = np.matmul(old_C_b_i, C_new_old)
    else:
        ave_C_b_i = old_C_b_i

    # Transform specific force to ECI-frame resolving axes using (5.81)
    f_ib_i = np.matmul(ave_C_b_i, f_ib_b)

    # UPDATE VELOCITY
    v_ib_i = old_v_ib_i + tor_i * (f_ib_i + gravitation_eci(old_r_ib_i))

    # UPDATE CARTESIAN POSITION
    # From (5.23)
    r_ib_i = old_r_ib_i + (v_ib_i + old_v_ib_i) * 0.5 * tor_i

    return r_ib_i, v_ib_i, C_b_i


def nav_equations_ecef(tor_i, old_r_eb_e, old_v_eb_e, old_C_b_e, f_ib_b, omega_ib_b):
    """
    Nav_equations_ECEF - Runs precision ECEF-frame inertial navigation
    equations

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 1/4/2012 by Paul Groves

    Inputs:
      tor_i         time interval between epochs (s)
      old_r_eb_e    previous Cartesian position of body frame w.r.t. ECEF
                    frame, resolved along ECEF-frame axes (m)
      old_C_b_e     previous body-to-ECEF-frame coordinate transformation matrix
      old_v_eb_e    previous velocity of body frame w.r.t. ECEF frame, resolved
                    along ECEF-frame axes (m/s)
      f_ib_b        specific force of body frame w.r.t. ECEF frame, resolved
                    along body-frame axes, averaged over time interval (m/s^2)
      omega_ib_b    angular rate of body frame w.r.t. ECEF frame, resolved
                    about body-frame axes, averaged over time interval (rad/s)
    Outputs:
      r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
                    along ECEF-frame axes (m)
      v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
                    ECEF-frame axes (m/s)
      C_b_e         body-to-ECEF-frame coordinate transformation matrix
    """

    # CONSTANT
    omega_ie = 7.292115e-5      # Earth rotation rate(rad / s)

    # ATTITUDE UPDATE
    # From (2.145) determine the Earth rotation over the update interval
    # C_Earth = C_e_i' * old_C_e_i
    alpha_ie = omega_ie * tor_i
    C_Earth = np.zeros((3, 3))
    C_Earth[0, 0] = math.cos(alpha_ie)
    C_Earth[0, 1] = math.sin(alpha_ie)
    C_Earth[1, 0] = -math.sin(alpha_ie)
    C_Earth[1, 1] = math.cos(alpha_ie)
    C_Earth[2, 2] = 1

    # Calculate attitude increment, magnitude, and skew-symmetric matrix
    alpha_ib_b = omega_ib_b * tor_i
    mag_alpha = math.sqrt(np.matmul(np.transpose(alpha_ib_b), alpha_ib_b))
    Alpha_ib_b = skew_symmetric(alpha_ib_b)

    # Obtain coordinate transformation matrix from the new attitude to the old
    # using Rodrigues' formula, (5.73)
    if mag_alpha > 1e-8:
        C_new_old = np.eye(3) + (math.sin(mag_alpha) / mag_alpha) * Alpha_ib_b + \
                    np.matmul(((1 - math.cos(mag_alpha)) / mag_alpha ** 2) * Alpha_ib_b, Alpha_ib_b)
    else:
        C_new_old = np.eye(3) + Alpha_ib_b

    # Update attitude using (5.75)
    C_b_e = np.matmul(np.matmul(C_Earth, old_C_b_e), C_new_old)

    # SPECIFIC FORCE FRAME TRANSFORMATION
    # Calculate the average body-to-ECEF-frame coordinate transformation
    # matrix over the update interval using (5.84) and (5.85)
    if mag_alpha > 1.e-8:
        first_order = ((1 - math.cos(mag_alpha)) / mag_alpha ** 2) * Alpha_ib_b
        second_order = ((1 - (math.sin(mag_alpha) / mag_alpha)) / mag_alpha ** 2) * np.matmul(Alpha_ib_b, Alpha_ib_b)
        C_new_old = np.identity(3) + first_order + second_order
        ave_C_b_e = np.matmul(old_C_b_e, C_new_old) - 0.5 * np.matmul(skew_symmetric(np.array([[0], [0], [alpha_ie]])), old_C_b_e)

    else:
        ave_C_b_e = old_C_b_e - 0.5 * np.matmul(skew_symmetric(np.array([[0], [0], [alpha_ie]])), old_C_b_e)

    # Transform specific force to ECEF-frame resolving axes using (5.85)
    f_ib_e = np.matmul(ave_C_b_e, f_ib_b)

    # UPDATE VELOCITY
    # From (5.36)
    v_eb_e = old_v_eb_e + tor_i * (f_ib_e + gravity_ecef(old_r_eb_e) - 2 * np.matmul(skew_symmetric(np.array([[0], [0], [omega_ie]])), old_v_eb_e))

    # UPDATE CARTESIAN POSITION
    # From (5.38)
    r_eb_e = old_r_eb_e + (v_eb_e + old_v_eb_e) * 0.5 * tor_i

    return r_eb_e, v_eb_e, C_b_e


def nav_equations_ned(tor_i, old_L_b, old_lambda_b, old_h_b, old_v_eb_n, old_C_b_n, f_ib_b, omega_ib_b):
    """
    Nav_equations_NED - Runs precision local-navigation-frame inertial
    navigation equations (Note: only the attitude update and specific force
    frame transformation phases are precise.)

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 1/4/2012 by Paul Groves

    Inputs:
      tor_i         time interval between epochs (s)
      old_L_b       previous latitude (rad)
      old_lambda_b  previous longitude (rad)
      old_h_b       previous height (m)
      old_C_b_n     previous body-to-NED coordinate transformation matrix
      old_v_eb_n    previous velocity of body frame w.r.t. ECEF frame, resolved
                    along north, east, and down (m/s)
      f_ib_b        specific force of body frame w.r.t. ECEF frame, resolved
                    along body-frame axes, averaged over time interval (m/s^2)
      omega_ib_b    angular rate of body frame w.r.t. ECEF frame, resolved
                    about body-frame axes, averaged over time interval (rad/s)
    Outputs:
      L_b           latitude (rad)
      lambda_b      longitude (rad)
      h_b           height (m)
      v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
                    north, east, and down (m/s)
      C_b_n         body-to-NED coordinate transformation matrix
    """

    # CONSTANT
    omega_ie = 7.292115e-5  # Earth rotation rate(rad / s)

    # PRELIMINARIES
    # Calculate attitude increment, magnitude, and skew-symmetric matrix
    alpha_ib_b = omega_ib_b * tor_i
    mag_alpha = math.sqrt(np.matmul(np.transpose(alpha_ib_b), alpha_ib_b))
    Alpha_ib_b = skew_symmetric(alpha_ib_b)

    # From (2.123) , determine the angular rate of the ECEF frame
    # w.r.t the ECI frame, resolved about NED
    omega_ie_n = omega_ie * np.array([[math.cos(old_L_b)], [0], [- math.sin(old_L_b)]])

    # From (5.44), determine the angular rate of the NED frame
    # w.r.t the ECEF frame, resolved about NED
    old_R_N, old_R_E = radii_of_curvature(old_L_b)
    old_omega_en_n = np.array([[old_v_eb_n[1, 0] / (old_R_E + old_h_b)], [-old_v_eb_n[0, 0] / (old_R_N + old_h_b)],
                               [-old_v_eb_n[1, 0] * math.tan(old_L_b) / (old_R_E + old_h_b)]])

    # SPECIFIC FORCE FRAME TRANSFORMATION
    # Calculate the average body-to-ECEF-frame coordinate transformation
    # matrix over the update interval using (5.84) and (5.86)
    if mag_alpha > 1.e-8:
        first_order = ((1 - math.cos(mag_alpha)) / mag_alpha ** 2) * Alpha_ib_b
        second_order = ((1 - (math.sin(mag_alpha) / mag_alpha)) / mag_alpha ** 2) * np.matmul(Alpha_ib_b, Alpha_ib_b)
        C_new_old = np.identity(3) + first_order + second_order
        ave_C_b_n = np.matmul(old_C_b_n, C_new_old) - 0.5 * np.matmul(skew_symmetric(old_omega_en_n + omega_ie_n), old_C_b_n)

    else:
        ave_C_b_n = old_C_b_n - 0.5 * np.matmul(skew_symmetric(old_omega_en_n + omega_ie_n), old_C_b_n)

    # Transform specific force to ECEF-frame resolving axes using (5.86)
    f_ib_n = np.matmul(ave_C_b_n, f_ib_b)

    # UPDATE VELOCITY
    # From (5.54)
    v_eb_n = old_v_eb_n + tor_i * (f_ib_n + gravity_ned(old_L_b, old_h_b) - np.matmul(skew_symmetric(old_omega_en_n + 2 * omega_ie_n), old_v_eb_n))

    # UPDATE CURVILINEAR POSITION
    # Update height using (5.56)
    h_b = old_h_b - 0.5 * tor_i * (old_v_eb_n[2, 0] + v_eb_n[2, 0])

    # Update latitude using (5.56)
    L_b = old_L_b + 0.5 * tor_i * (old_v_eb_n[0, 0] / (old_R_N + old_h_b) + v_eb_n[0, 0] / (old_R_N + h_b))

    # Calculate meridian and transverse radii of curvature
    R_N, R_E= radii_of_curvature(L_b)

    # Update longitude using (5.56)
    lambda_b = old_lambda_b + 0.5 * tor_i * (old_v_eb_n[1, 0] / ((old_R_E + old_h_b) * math.cos(old_L_b)) + v_eb_n[1, 0] / ((R_E + h_b) * math.cos(L_b)))

    # ATTITUDE UPDATE
    # From (5.44), determine the angular rate of the NED frame
    # w.r.t the ECEF frame, resolved about NED
    omega_en_n = np.array([[v_eb_n[1, 0] / (R_E + h_b)], [- v_eb_n[0, 0] / (R_N + h_b)], [- v_eb_n[1, 0] * math.tan(L_b) / (R_E + h_b)]])

    # Obtain coordinate transformation matrix from the new attitude w.r.t. an
    # inertial frame to the old using Rodrigues' formula, (5.73)
    if mag_alpha > 1e-8:
        C_new_old = np.identity(3) + (math.sin(mag_alpha) / mag_alpha) * Alpha_ib_b + \
                    ((1 - math.cos(mag_alpha)) / (mag_alpha ** 2)) * np.matmul(Alpha_ib_b, Alpha_ib_b)
    else:
        C_new_old = np.identity(3) + Alpha_ib_b

    # Update attitude using (5.77)
    C_b_n = np.matmul(np.matmul((np.identity(3) - skew_symmetric(omega_ie_n + 0.5 * omega_en_n + 0.5 * old_omega_en_n) * tor_i), old_C_b_n), C_new_old)

    return L_b, lambda_b, h_b, v_eb_n, C_b_n


