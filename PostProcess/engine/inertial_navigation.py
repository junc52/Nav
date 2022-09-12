import math
import numpy as np
import quaternion

from PostProcess.utils.frame_transform import skew_symmetric

INS_MESSAGE_TYPE = ['UncalAccel', 'UncalGyro', 'UncalMag']
MS_TO_SEC = 1 / 1000


class NavIns:
    def __init__(self, args):
        self._args = args
        self._utc_time_millis_a = 0
        self._last_utc_time_millis_a = 0
        self._utc_time_millis_g = 0
        self._last_utc_time_millis_g = 0
        self._f_ib_b = np.zeros((3, 1))
        self._omega_ib_b = np.zeros((3, 1))
        self._alpha_ib_b = np.zeros((3, 1))  # attitude increment (rad)
        self._C_new_old = None
        self._C_b_i = None

    @property
    def f_ib_b(self):
        return self._f_ib_b

    @property
    def omega_ib_b(self):
        return self._omega_ib_b

    @f_ib_b.setter
    def f_ib_b(self, val):
        self._f_ib_b = val.to_numpy().reshape((3, 1))

    @omega_ib_b.setter
    def omega_ib_b(self, val):
        self._omega_ib_b = val.to_numpy().reshape((3, 1))

    def get_attitude_update_matrix(self):
        t_interval = (self._utc_time_millis_g - self._last_utc_time_millis_g) * MS_TO_SEC

        # Calculate attitude increment, magnitude, and skew-symmetric matrix
        alpha_ib_b = self._omega_ib_b * t_interval
        mag_alpha = np.linalg.norm(alpha_ib_b)
        Alpha_ib_b = skew_symmetric(alpha_ib_b)

        # Obtain coordinate transformation matrix from the new attitude to the old
        # using Rodrigues' formula, Paul Groves. (5.73)
        if mag_alpha > 1e-8:
            first_order = (math.sin(mag_alpha) / mag_alpha) * Alpha_ib_b
            second_order = ((1 - math.cos(mag_alpha)) / mag_alpha ** 2) * np.matmul(Alpha_ib_b, Alpha_ib_b)
            self._C_new_old = np.eye(3) + first_order + second_order
        else:
            self._C_new_old = np.eye(3) + Alpha_ib_b

    def update_attitude(self):
        raise NotImplementedError('virtual method specification of base class')


class NavInsEci(NavIns):
    def update_attitude(self):
        self.get_attitude_update_matrix()

        # TODO 1: change to quaternion representation
        # TODO 2: add coordinate transformation matrix initialization logic
        self._C_b_i = np.matmul(self._C_b_i, self._C_new_old)


