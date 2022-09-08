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
# Read_profile - inputs a motion profile in the following .csv format
# Column 1: time (sec)
# Column 2: latitude (deg)
# Column 3: longitude (deg)
# Column 4: height (m)
# Column 5: north velocity (m/s)
# Column 6: east velocity (m/s)
# Column 7: down velocity (m/s)
# Column 8: roll angle of body w.r.t NED (deg)
# Column 9: pitch angle of body w.r.t NED (deg)
# Column 10: yaw angle of body w.r.t NED (deg)
#
# Software for use with "Principles of GNSS, Inertial, and Multisensor
# Integrated Navigation Systems," Second Edition.
#
# This function created 31/3/2012 by Paul Groves
#
# Inputs:
#   filename     Name of file to write
#
# Outputs:
#   in_profile   Array of data from the file
#   no_epochs    Number of epochs of data in the file
#   ok           Indicates file has the expected number of columns

# Copyright 2012, Paul Groves
# License: BSD; see license.txt for details
from pandas import read_csv

# CONSTANT
DEG_TO_RAD = 0.01745329252


def read_profile(input_profile_name):
    # Read in the profile in.csv format
    df = read_csv(input_profile_name)
    in_profile = df.values

    # Check number of columns is correct (otherwise return)
    if in_profile.shape[1] != 10:
        print("Input file has the wrong number of columns")
    else:
        # Convert degrees to radian
        in_profile[:, 1:3] = DEG_TO_RAD * in_profile[:, 1:3]
        in_profile[:, 7:] = DEG_TO_RAD * in_profile[:, 7:]

    return in_profile

