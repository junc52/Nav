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
#  Copyright 2012, Paul Groves
#  License: BSD; see license.txt for details
#

import math
import numpy as np


def skew_symmetric(a):
    """
    Skew_symmetric - Calculates skew-symmetric matrix

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 1/4/2012 by Paul Groves

    Inputs:
        a       3-element vector
    Outputs:
        A       3x3matrix
    """
    A = np.zeros((3, 3))
    A[0, 1] = -a[2, 0]
    A[0, 2] = a[1, 0]
    A[1, 0] = a[2, 0]
    A[1, 2] = -a[0, 0]
    A[2, 0] = -a[1, 0]
    A[2, 1] = a[0, 0]

    return A


def euler_to_ctm(eul):
    """
    Euler_to_CTM - Converts a set of Euler angles to the corresponding
    coordinate transformation matrix

    This function created 1/4/2012 by Paul Groves

    Inputs:
      eul     Euler angles describing rotation from beta to alpha in the
              order roll, pitch, yaw(rad)

    Outputs:
      C       coordinate transformation matrix describing transformation from
              beta to alpha
    """
    # Precalculate sines and cosines of the Euler angles
    sin_phi   = math.sin(eul[0, 0])
    cos_phi   = math.cos(eul[0, 0])
    sin_theta = math.sin(eul[1, 0])
    cos_theta = math.cos(eul[1, 0])
    sin_psi   = math.sin(eul[2, 0])
    cos_psi   = math.cos(eul[2, 0])

    # Calculate coordinate transformation matrix using (2.22)
    C = np.zeros((3, 3))
    C[0, 0] = cos_theta * cos_psi
    C[0, 1] = cos_theta * sin_psi
    C[0, 2] = -sin_theta
    C[1, 0] = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
    C[1, 1] = cos_phi * cos_psi + sin_phi * sin_theta * sin_psi
    C[1, 2] = sin_phi * cos_theta
    C[2, 0] = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi
    C[2, 1] = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi
    C[2, 2] = cos_phi * cos_theta

    return C


def ctm_to_euler(C):
    """
    CTM_to_Euler - Converts a coordinate transformation matrix to the
    corresponding set of Euler angles%

    This function created 1/4/2012 by Paul Groves

    Inputs:
      C       coordinate transformation matrix describing transformation from
              beta to alpha

    Outputs:
      eul     Euler angles describing rotation from beta to alpha in the
              order roll, pitch, yaw(rad)
    """
    # Calculate Euler angles using (2.23)
    eul = np.zeros((3, 1))
    eul[0, 0] = math.atan2(C[1, 2], C[2, 2])  # roll
    eul[1, 0] = - math.asin(C[0, 2])          # pitch
    eul[2, 0] = math.atan2(C[0, 1], C[0, 0])  # yaw

    return eul


def eci_to_ecef(t,r_ib_i,v_ib_i,C_b_i):
    """
    ECI_to_ECEF - Converts position, velocity, and attitude from ECI- to
    ECEF-frame referenced and resolved

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 2/4/2012 by Paul Groves

    Inputs:
      t             time (s)
      r_ib_i        Cartesian position of body frame w.r.t. ECI frame, resolved
                    along ECI-frame axes (m)
      v_ib_i        velocity of body frame w.r.t. ECI frame, resolved along
                    ECI-frame axes (m/s)
      C_b_i         body-to-ECI-frame coordinate transformation matrix

    Outputs:
      r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
                    along ECEF-frame axes (m)
      v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
                    ECEF-frame axes (m/s)
      C_b_e         body-to-ECEF-frame coordinate transformation matrix
    """
    # CONSTANTS
    omega_ie = 7.292115e-5  # Earth rotation rate (rad/s)

    # Calculate ECI to ECEF coordinate transformation matrix using (2.145)
    C_i_e = np.zeros((3, 3))
    C_i_e[0, 0] = math.cos(omega_ie * t)
    C_i_e[0, 1] = math.sin(omega_ie * t)
    C_i_e[1, 0] = -math.sin(omega_ie * t)
    C_i_e[1, 1] = math.cos(omega_ie * t)
    C_i_e[2, 2] = 1

    # Transform position using (2.146)
    r_eb_e = np.matmul(C_i_e, r_ib_i)

    # Transform velocity using (2.145)
    v_eb_e = np.matmul(C_i_e, (v_ib_i - omega_ie * np.array([[-r_ib_i[1, 0]], [r_ib_i[0, 0]], [0]])))

    # Transform attitude using (2.15)
    C_b_e = np.matmul(C_i_e, C_b_i)

    return r_eb_e, v_eb_e, C_b_e


def ecef_to_eci(t, r_eb_e, v_eb_e, C_b_e):
    """
    ECEF_to_ECI - Converts position, velocity, and attitude from ECEF- to
    ECI-frame referenced and resolved

    Inputs:
      t             time (s)
      r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
                    along ECEF-frame axes (m)
      v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
                    ECEF-frame axes (m/s)
      C_b_e         body-to-ECEF-frame coordinate transformation matrix

    Outputs:
      r_ib_i        Cartesian position of body frame w.r.t. ECI frame, resolved
                    along ECI-frame axes (m)
      v_ib_i        velocity of body frame w.r.t. ECI frame, resolved along
                    ECI-frame axes (m/s)
      C_b_i         body-to-ECI-frame coordinate transformation matrix
    """
    # CONSTANTS
    omega_ie = 7.292115e-5          # Earth rotation rate (rad/s)

    # Calculate ECEF to ECI coordinate transformation matrix using (2.145)
    C_e_i = np.zeros((3, 3))
    C_e_i[0, 0] = math.cos(omega_ie * t)
    C_e_i[0, 1] = -math.sin(omega_ie * t)
    C_e_i[1, 0] = math.sin(omega_ie * t)
    C_e_i[1, 1] = math.cos(omega_ie * t)
    C_e_i[2, 2] = 1

    # Transform position using (2.146)
    r_ib_i = np.matmul(C_e_i, r_eb_e)

    # Transform velocity using (2.147)
    v_ib_i = np.matmul(C_e_i, (v_eb_e + omega_ie * np.array([[-r_eb_e[1, 0]], [r_eb_e[0, 0]], [0]])))

    # Transform attitude using (2.15)
    C_b_i = np.matmul(C_e_i, C_b_e)

    return r_ib_i, v_ib_i, C_b_i


def ecef_to_ned(r_eb_e,v_eb_e,C_b_e):
    """
    ECEF_to_NED - Converts Cartesian  to curvilinear position, velocity
    resolving axes from ECEF to NED and attitude from ECEF- to NED-referenced

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 2/4/2012 by Paul Groves

    Inputs:
      r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
                    along ECEF-frame axes (m)
      v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
                    ECEF-frame axes (m/s)
      C_b_e         body-to-ECEF-frame coordinate transformation matrix

    Outputs:
      L_b           latitude (rad)
      lambda_b      longitude (rad)
      h_b           height (m)
      v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
                    north, east, and down (m/s)
      C_b_n         body-to-NED coordinate transformation matrix
    """
    # CONSTANTS
    R_0 = 6378137  # WGS84 Equatorial radius in meters
    e = 0.0818191908425  # WGS84 eccentricity

    # Convert position using Borkowski closed-form exact solution
    # From (2.113)
    lambda_b = math.atan2(r_eb_e[1, 0], r_eb_e[0, 0])

    # From (C.29) and (C.30)
    k1 = math.sqrt(1 - e ** 2) * abs(r_eb_e[2, 0])
    k2 = e ** 2 * R_0
    beta = math.sqrt(r_eb_e[0, 0] ** 2 + r_eb_e[1, 0] ** 2)
    E = (k1 - k2) / beta
    F = (k1 + k2) / beta

    # From (C.31)
    P = (4 / 3) * (E * F + 1)

    # From (C.32)
    Q = 2 * (E ** 2 - F ** 2)

    # From (C.33)
    D = P ** 3 + Q ** 2

    # From (C.34)
    V = (math.sqrt(D) - Q) ** (1 / 3) - (math.sqrt(D) + Q) ** (1 / 3)

    # From (C.35)
    G = 0.5 * (math.sqrt(E ** 2 + V) + E)

    # From (C.36)
    T = math.sqrt(G ** 2 + (F - V * G) / (2 * G - E)) - G

    # From (C.37)
    L_b = np.sign(r_eb_e[2, 0]) * math.atan((1 - T ** 2) / (2 * T * math.sqrt(1 - e ** 2)))

    # From (C.38)
    h_b = (beta - R_0 * T) * math.cos(L_b) + (r_eb_e[2, 0] - np.sign(r_eb_e[2, 0]) * R_0 * math.sqrt(1 - e ** 2)) * math.sin(L_b)

    # Calculate ECEF to NED coordinate transformation matrix using (2.150)
    cos_lat = math.cos(L_b)
    sin_lat = math.sin(L_b)
    cos_long = math.cos(lambda_b)
    sin_long = math.sin(lambda_b)
    C_e_n = np.zeros((3, 3))
    C_e_n[0, 0] = -sin_lat * cos_long
    C_e_n[0, 1] = -sin_lat * sin_long
    C_e_n[0, 2] = cos_lat
    C_e_n[1, 0] = -sin_long
    C_e_n[1, 1] = cos_long
    C_e_n[2, 0] = -cos_lat * cos_long
    C_e_n[2, 1] = -cos_lat * sin_long
    C_e_n[2, 2] = -sin_lat

    # Transform velocity using (2.73)
    v_eb_n = np.matmul(C_e_n, v_eb_e)

    # Transform attitude using (2.15)
    C_b_n = np.matmul(C_e_n, C_b_e)

    return L_b, lambda_b, h_b, v_eb_n, C_b_n


def ned_to_ecef(L_b, lambda_b, h_b, v_eb_n, C_b_n):
    """
    NED_to_ECEF - Converts curvilinear to Cartesian position, velocity
    resolving axes from NED to ECEF and attitude from NED- to ECEF-referenced

    This function created 2/4/2012 by Paul Groves

    Inputs:
      L_b           latitude (rad)
      lambda_b      longitude (rad)
      h_b           height (m)
      v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
                    north, east, and down (m/s)
      C_b_n         body-to-NED coordinate transformation matrix

    Outputs:
      r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
                    along ECEF-frame axes (m)
      v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
                    ECEF-frame axes (m/s)
      C_b_e         body-to-ECEF-frame coordinate transformation matrix
    """

    # CONSTANTS
    R_0 = 6378137           # WGS84 Equatorial radius in meters
    e = 0.0818191908425     # WGS84 eccentricity

    # Calculate transverse radius of curvature using (2.106)
    R_E = R_0 / math.sqrt(1 - (e * math.sin(L_b)) ** 2)

    # Convert position using (2.112)
    r_eb_e       = np.zeros((3, 1))
    cos_lat      = math.cos(L_b)
    sin_lat      = math.sin(L_b)
    cos_long     = math.cos(lambda_b)
    sin_long     = math.sin(lambda_b)
    r_eb_e[0, 0] = (R_E + h_b) * cos_lat * cos_long
    r_eb_e[1, 0] = (R_E + h_b) * cos_lat * sin_long
    r_eb_e[2, 0] = ((1 - e ** 2) * R_E + h_b) * sin_lat

    # Calculate ECEF to NED coordinate transformation matrix using (2.150)
    C_e_n = np.zeros((3, 3))
    C_e_n[0, 0] = -sin_lat * cos_long
    C_e_n[0, 1] = -sin_lat * sin_long
    C_e_n[0, 2] = cos_lat
    C_e_n[1, 0] = -sin_long
    C_e_n[1, 1] = cos_long
    C_e_n[2, 0] = -cos_lat * cos_long
    C_e_n[2, 1] = -cos_lat * sin_long
    C_e_n[2, 2] = -sin_lat

    # Transform velocity using (2.73)
    v_eb_e = np.matmul(np.transpose(C_e_n), v_eb_n)

    # Transform attitude using (2.15)
    C_b_e = np.matmul(np.transpose(C_e_n), C_b_n)

    return r_eb_e, v_eb_e, C_b_e


def pv_ecef_to_ned(r_eb_e,v_eb_e):
    """
    pv_ECEF_to_NED - Converts Cartesian  to curvilinear position, velocity
    resolving axes from ECEF to NED

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 2/4/2012 by Paul Groves

    Inputs:
      r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
                    along ECEF-frame axes (m)
      v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
                    ECEF-frame axes (m/s)

    Outputs:
      L_b           latitude (rad)
      lambda_b      longitude (rad)
      h_b           height (m)
      v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
                    north, east, and down (m/s)
    """
    # CONSTANTS
    R_0 = 6378137  # WGS84 Equatorial radius in meters
    e = 0.0818191908425  # WGS84 eccentricity

    # Convert position using Borkowski closed-form exact solution
    # From (2.113)
    lambda_b = math.atan2(r_eb_e[1, 0], r_eb_e[0, 0])

    # From (C.29) and (C.30)
    k1 = math.sqrt(1 - e ** 2) * abs(r_eb_e[2, 0])
    k2 = e ** 2 * R_0
    beta = math.sqrt(r_eb_e[0, 0] ** 2 + r_eb_e[1, 0] ** 2)
    E = (k1 - k2) / beta
    F = (k1 + k2) / beta

    # From (C.31)
    P = (4 / 3) * (E * F + 1)

    # From (C.32)
    Q = 2 * (E ** 2 - F ** 2)

    # From (C.33)
    D = P ** 3 + Q ** 2

    # From (C.34)
    V = (math.sqrt(D) - Q) ** (1 / 3) - (math.sqrt(D) + Q) ** (1 / 3)

    # From (C.35)
    G = 0.5 * (math.sqrt(E ** 2 + V) + E)

    # From (C.36)
    T = math.sqrt(G ** 2 + (F - V * G) / (2 * G - E)) - G

    # From (C.37)
    L_b = np.sign(r_eb_e[2, 0]) * math.atan((1 - T ** 2) / (2 * T * math.sqrt(1 - e ** 2)))

    # From (C.38)
    h_b = (beta - R_0 * T) * math.cos(L_b) + (r_eb_e[2, 0] - np.sign(r_eb_e[2, 0]) * R_0 * math.sqrt(1 - e ** 2)) * math.sin(L_b)

    # Calculate ECEF to NED coordinate transformation matrix using (2.150)
    cos_lat = math.cos(L_b)
    sin_lat = math.sin(L_b)
    cos_long = math.cos(lambda_b)
    sin_long = math.sin(lambda_b)
    C_e_n = np.zeros((3, 3))
    C_e_n[0, 0] = -sin_lat * cos_long
    C_e_n[0, 1] = -sin_lat * sin_long
    C_e_n[0, 2] = cos_lat
    C_e_n[1, 0] = -sin_long
    C_e_n[1, 1] = cos_long
    C_e_n[2, 0] = -cos_lat * cos_long
    C_e_n[2, 1] = -cos_lat * sin_long
    C_e_n[2, 2] = -sin_lat

    # Transform velocity using (2.73)
    v_eb_n = np.matmul(C_e_n, v_eb_e)

    return L_b, lambda_b, h_b, v_eb_n


def pv_ned_to_ecef(L_b, lambda_b, h_b, v_eb_n):
    """
    pv_NED_to_ECEF - Converts curvilinear to Cartesian position and velocity
    resolving axes from NED to ECEF

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves

    Inputs:
      L_b           latitude (rad)
      lambda_b      longitude (rad)
      h_b           height (m)
      v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
                    north, east, and down (m/s)

    Outputs:
      r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
                    along ECEF-frame axes (m)
      v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
                    ECEF-frame axes (m/s)

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """

    # CONSTANTS
    R_0 = 6378137           # WGS84 Equatorial radius in meters
    e = 0.0818191908425     # WGS84 eccentricity

    # Calculate transverse radius of curvature using (2.106)
    R_E = R_0 / math.sqrt(1 - (e * math.sin(L_b)) ** 2)

    # Convert position using (2.112)
    r_eb_e = np.zeros((3, 1))
    cos_lat = math.cos(L_b)
    sin_lat = math.sin(L_b)
    cos_long = math.cos(lambda_b)
    sin_long = math.sin(lambda_b)
    r_eb_e[0, 0] = (R_E + h_b) * cos_lat * cos_long
    r_eb_e[1, 0] = (R_E + h_b) * cos_lat * sin_long
    r_eb_e[2, 0] = ((1 - e ** 2) * R_E + h_b) * sin_lat

    # Calculate ECEF to NED coordinate transformation matrix using (2.150)
    C_e_n = np.zeros((3, 3))
    C_e_n[0, 0] = -sin_lat * cos_long
    C_e_n[0, 1] = -sin_lat * sin_long
    C_e_n[0, 2] = cos_lat
    C_e_n[1, 0] = -sin_long
    C_e_n[1, 1] = cos_long
    C_e_n[2, 0] = -cos_lat * cos_long
    C_e_n[2, 1] = -cos_lat * sin_long
    C_e_n[2, 2] = -sin_lat

    # Transform velocity using (2.73)
    v_eb_e = np.matmul(np.transpose(C_e_n), v_eb_n)

    return r_eb_e, v_eb_e


