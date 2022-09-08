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


def radii_of_curvature(L):
    """
    Radii_of_curvature - Calculates the meridian and transverse radii of
    curvature

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 31/3/2012 by Paul Groves

    Inputs:
      L   geodetic latitude (rad)

    Outputs:
      R_N   meridian radius of curvature (m)
      R_E   transverse radius of curvature (m)
    """
    # CONSTANTS
    R_0 = 6378137           # WGS84 Equatorial radius in meters
    e = 0.0818191908425     # WGS84 eccentricity

    # Calculate meridian radius of curvature using (2.105)
    temp = 1 - (e * math.sin(L)) ** 2
    R_N = R_0 * (1 - e ** 2) / temp ** 1.5

    # Calculate transverse radius of curvature using (2.105)
    R_E = R_0 / math.sqrt(temp)

    return R_N, R_E