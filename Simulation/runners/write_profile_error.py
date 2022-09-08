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

import pandas as pd


def write_profile(filename, out_profile):
    """
    Write_profile - outputs a motion profile in the following .csv format
    Column 1: time (sec)
    Column 2: latitude (deg)
    Column 3: longitude (deg)
    Column 4: height (m)
    Column 5: north velocity (m/s)
    Column 6: east velocity (m/s)
    Column 7: down velocity (m/s)
    Column 8: roll angle of body w.r.t NED (deg)
    Column 9: pitch angle of body w.r.t NED (deg)
    Column 10: yaw angle of body w.r.t NED (deg)

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 31/3/2012 by Paul Groves

    Inputs:
      filename     Name of file to write
      out_profile  Array of data to write

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """

    # CONSTANTS
    DEG_TO_RAD = 0.01745329252
    RAD_TO_DEG = 1 / DEG_TO_RAD

    # Convert output profile from radians to degrees
    out_profile[:, 1: 3] = RAD_TO_DEG * out_profile[:, 1: 3]
    out_profile[:, 7: 10] = RAD_TO_DEG * out_profile[:, 7: 10]

    # Write output profile
    pd.DataFrame(out_profile).to_csv(filename)


def write_errors(filename, out_errors):
    """
    Write_errors - outputs the errors in the following .csv format
    Column 1: time (sec)
    Column 2: north position error (m)
    Column 3: east position error (m)
    Column 4: down position error (m)
    Column 5: north velocity (m/s)
    Column 6: east velocity (m/s)
    Column 7: down velocity (m/s)
    Column 8: roll component of NED attitude error (deg)
    Column 9: pitch component of NED attitude error (deg)
    Column 10: yaw component of NED attitude error (deg)

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 31/3/2012 by Paul Groves

    Inputs:
      filename     Name of file to write
      out_errors   Array of data to write

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """

    # CONSTANTS
    DEG_TO_RAD = 0.01745329252
    RAD_TO_DEG = 1 / DEG_TO_RAD

    # Convert output profile from radians to degrees
    out_errors[:, 7: 10] = RAD_TO_DEG * out_errors[:, 7: 10]

    # Write output profile
    pd.DataFrame(out_errors).to_csv(filename)
