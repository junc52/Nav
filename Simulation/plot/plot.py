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

import math
import matplotlib.pyplot as plt
from Simulation.utils.curvilinear_conversion import radii_of_curvature


def plot_profile(profile):
    """
    Plots a motion profile

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 3/4/2012 by Paul Groves

    Input:
      profile      Array of motion profile data to plot
    Format is
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

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """

    DEG_TO_RAD = 0.01745329252
    RAD_TO_DEG = 1 / DEG_TO_RAD

    R_N, R_E = radii_of_curvature(profile[0, 1])
    fig, axes = plt.subplots(3, 3)

    # North Displacement profile
    x = profile[:, [0]]
    y = (profile[:, [1]] - profile[0, 1]) * (R_N + profile[0, 3])
    axes[0, 0].plot(x, y)
    axes[0, 0].set_xlabel('time(s)')
    axes[0, 0].set_ylabel('m')
    axes[0, 0].set_title('North displacement, m')
    axes[0, 0].grid(True)

    # East Displacement profile
    y = (profile[:, [2]] - profile[0, 2]) * (R_N + profile[0, 3]) * math.cos(profile[0, 1])
    axes[0, 1].plot(x, y)
    axes[0, 1].set_xlabel('time(s)')
    axes[0, 1].set_ylabel('m')
    axes[0, 1].set_title('East displacement, m')
    axes[0, 1].grid(True)

    # Down Displacement profile
    y = (profile[0, 3] - profile[:, [3]])
    axes[0, 2].plot(x, y)
    axes[0, 2].set_xlabel('time(s)')
    axes[0, 2].set_ylabel('m')
    axes[0, 2].set_title('Down displacement, m')
    axes[0, 2].grid(True)

    # Notrh Velocity(m/s) profile
    y = profile[:, [4]]
    axes[1, 0].plot(x, y)
    axes[1, 0].set_xlabel('time(s)')
    axes[1, 0].set_ylabel('m/s')
    axes[1, 0].set_title('North Velocity, m/s')
    axes[1, 0].grid(True)

    # East Velocity(m/s) profile
    y = profile[:, [5]]
    axes[1, 1].plot(x, y)
    axes[1, 1].set_xlabel('time(s)')
    axes[1, 1].set_ylabel('m/s')
    axes[1, 1].set_title('East Velocity, m/s')
    axes[1, 1].grid(True)

    # Down Velocity(m/s) profile
    y = profile[:, [6]]
    axes[1, 2].plot(x, y)
    axes[1, 2].set_xlabel('time(s)')
    axes[1, 2].set_ylabel('m/s')
    axes[1, 2].set_title('North Velocity, m/s')
    axes[1, 2].grid(True)

    # Roll  profile
    y = profile[:, [7]] * RAD_TO_DEG
    axes[2, 0].plot(x, y)
    axes[2, 0].set_xlabel('time(s)')
    axes[2, 0].set_ylabel('deg')
    axes[2, 0].set_title('Roll, deg')
    axes[2, 0].grid(True)

    # Pitch  profile
    y = profile[:, [8]] * RAD_TO_DEG
    axes[2, 1].plot(x, y)
    axes[2, 1].set_xlabel('time(s)')
    axes[2, 1].set_ylabel('deg')
    axes[2, 1].set_title('Pitch, deg')
    axes[2, 1].grid(True)

    # Yaw  profile
    y = profile[:, [9]] * RAD_TO_DEG
    axes[2, 2].plot(x, y)
    axes[2, 2].set_xlabel('time(s)')
    axes[2, 2].set_ylabel('deg')
    axes[2, 2].set_title('Yaw, deg')
    axes[2, 2].grid(True)

    plt.show()


def plot_errors(errors):
    """
    Plots navigation solution errors

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 3/4/2012 by Paul Groves

    Input:
      errors      Array of error data to plot
    Format is
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

    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """

    DEG_TO_RAD = 0.01745329252
    RAD_TO_DEG = 1 / DEG_TO_RAD

    fig, axes = plt.subplots(3, 3)

    # North Position Error
    x = errors[:, [0]]
    y = errors[:, [1]]
    axes[0, 0].plot(x, y)
    axes[0, 0].set_xlabel('time(s)')
    axes[0, 0].set_ylabel('m')
    axes[0, 0].set_title('North Position Error, m')
    axes[0, 0].grid(True)

    # East Position Error
    y = errors[:, [2]]
    axes[0, 1].plot(x, y)
    axes[0, 1].set_xlabel('time(s)')
    axes[0, 1].set_ylabel('m')
    axes[0, 1].set_title('East Position Error, m')
    axes[0, 1].grid(True)

    # Down Position Error
    y = errors[:, [3]]
    axes[0, 2].plot(x, y)
    axes[0, 2].set_xlabel('time(s)')
    axes[0, 2].set_ylabel('m')
    axes[0, 2].set_title('Down Position Error, m')
    axes[0, 2].grid(True)

    # Notrh Velocity(m/s) Error
    y = errors[:, [4]]
    axes[1, 0].plot(x, y)
    axes[1, 0].set_xlabel('time(s)')
    axes[1, 0].set_ylabel('m/s')
    axes[1, 0].set_title('North Velocity Error, m/s')
    axes[1, 0].grid(True)

    # East Velocity(m/s) Error
    y = errors[:, [5]]
    axes[1, 1].plot(x, y)
    axes[1, 1].set_xlabel('time(s)')
    axes[1, 1].set_ylabel('m/s')
    axes[1, 1].set_title('East Velocity Error, m/s')
    axes[1, 1].grid(True)

    # Down Velocity(m/s) Error
    y = errors[:, [6]]
    axes[1, 2].plot(x, y)
    axes[1, 2].set_xlabel('time(s)')
    axes[1, 2].set_ylabel('m/s')
    axes[1, 2].set_title('North Velocity Error, m/s')
    axes[1, 2].grid(True)

    # Attitude error about North
    y = errors[:, [7]] * RAD_TO_DEG
    axes[2, 0].plot(x, y)
    axes[2, 0].set_xlabel('time(s)')
    axes[2, 0].set_ylabel('deg')
    axes[2, 0].set_title('Attitude error about North, deg')
    axes[2, 0].grid(True)

    # Attitude error about East
    y = errors[:, [8]] * RAD_TO_DEG
    axes[2, 1].plot(x, y)
    axes[2, 1].set_xlabel('time(s)')
    axes[2, 1].set_ylabel('deg')
    axes[2, 1].set_title('Attitude error about East, deg')
    axes[2, 1].grid(True)

    # Heading error
    y = errors[:, [9]] * RAD_TO_DEG
    axes[2, 2].plot(x, y)
    axes[2, 2].set_xlabel('time(s)')
    axes[2, 2].set_ylabel('deg')
    axes[2, 2].set_title('Heading error, deg')
    axes[2, 2].grid(True)

    plt.show()
