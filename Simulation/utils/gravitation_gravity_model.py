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


def gravitation_eci(r_ib_i):
    """
    Gravitation_ECI - Calculates gravitational acceleration resolved about
    ECI-frame axes

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 1/4/2012 by Paul Groves

    Inputs:
      r_ib_i  Cartesian position of body frame w.r.t. ECI frame, resolved
              about ECI-frame axes (m)
    Outputs:
      gamma   Acceleration due to gravitational force (m/s^2)
    """

    # CONSTANTS
    R_0 = 6378137           # WGS84 Equatorial radius in meters
    mu = 3.986004418E14     # WGS84 Earth gravitational constant (m^3 s^-2)
    J_2 = 1.082627e-3       # WGS84 Earth's second gravitational constant

    # (2.141) models the gravitational acceleration
    # Calculate distance from center of the Earth
    mag_r = np.linalg.norm(r_ib_i)

    # if the input position is 0,0,0, produce a dummy output
    if mag_r == 0:
        gamma = np.zeros((3, 1))

    # Calculate gravitational acceleration using (2.141)
    else:
        z_scale = 5 * ((r_ib_i[2, 0] / mag_r) ** 2)
        gamma = -mu / (mag_r ** 3) * (r_ib_i + 1.5 * J_2 * ((R_0 / mag_r) ** 2) *
                np.array([[(1 - z_scale) * r_ib_i[0, 0]], [(1 - z_scale) * r_ib_i[1, 0]],
                          [(3 - z_scale) * r_ib_i[2, 0]]]))

    return gamma


def gravity_ecef(r_eb_e):
    """
    Gravitation_ECI - Calculates  acceleration due to gravity resolved about ECEF-frame

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 1/4/2012 by Paul Groves

    Inputs:
      r_eb_e  Cartesian position of body frame w.r.t. ECEF frame, resolved
              about ECEF-frame axes (m)
    Outputs:
      g       Acceleration due to gravity (m/s^2)
    """
    # CONSTANTS
    R_0 = 6378137           # WGS84 Equatorial radius in meters
    mu = 3.986004418E14     # WGS84 Earth gravitational constant(m ^ 3 s ^ -2)
    J_2 = 1.082627e-3       # WGS84 Earth's second gravitational constant
    omega_ie = 7.292115e-5  # Earth rotation rate(rad / s)

    # Calculate distance from center of the Earth
    mag_r = np.linalg.norm(r_eb_e)

    # If the input position is 0,0,0, produce a dummy output
    g = np.zeros((3, 1))
    if mag_r != 0:
        z_scale = 5 * (r_eb_e[2, 0] / mag_r) ** 2
        gamma = -mu / mag_r ** 3 * (r_eb_e + 1.5 * J_2 * (R_0 / mag_r) ** 2 *
                                    np.array([[(1 - z_scale) * r_eb_e[0, 0]], [(1 - z_scale) * r_eb_e[1, 0]],
                                              [(3 - z_scale) * r_eb_e[2, 0]]]))

        # Add centripetal acceleration using (2.133)
        g[0:2, [0]] = gamma[0:2, [0]] + omega_ie ** 2 * r_eb_e[0:2, [0]]
        g[2, 0] = gamma[2, 0]

    return g


def gravity_ned(L_b,h_b):
    """
    Gravity_ECEF - Calculates  acceleration due to gravity resolved about
    north, east, and down

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 2/4/2012 by Paul Groves

    Inputs:
      L_b           latitude (rad)
      h_b           height (m)
    Outputs:
      g       Acceleration due to gravity (m/s^2)
    """
    # CONSTANTS
    R_0 = 6378137           # WGS84 Equatorial radius in meters
    R_P = 6356752.31425     # WGS84 Polar radius in meters
    e = 0.0818191908425     # WGS84 eccentricity
    f = 1 / 298.257223563   # WGS84 flattening
    mu = 3.986004418e14     # WGS84 Earth gravitational constant(m ^ 3 s ^ -2)
    omega_ie = 7.292115e-5  # Earth rotation rate(rad / s)

    # Calculate surface gravity using the Somigliana model, (2.134)
    sinsqL = math.sin(L_b) ** 2
    g_0 = 9.7803253359 * (1 + 0.001931853 * sinsqL) / math.sqrt(1 - e ** 2 * sinsqL)

    # Calculate north gravity using (2.140)
    g = np.zeros((3, 1))
    g[0, 0] = -8.08E-9 * h_b * math.sin(2 * L_b)

    # East gravity is zero
    g[1, 0] = 0
    # Calculate down gravity using (2.139)
    g[2, 0] = g_0 * (1 - (2 / R_0) * (1 + f * (1 - 2 * sinsqL) + (omega_ie ** 2 * R_0 ** 2 * R_P / mu)) *
                     h_b + (3 * h_b ** 2 / R_0 ** 2))

    return g
