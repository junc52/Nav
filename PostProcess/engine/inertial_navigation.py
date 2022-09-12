import math
import numpy as np
import quaternion

from PostProcess.utils.frame_transform import skew_symmetric

INS_MESSAGE_TYPE = ['UncalAccel', 'UncalGyro', 'UncalMag']
MS_TO_SEC = 1 / 1000


class NavIns:
    def __init__(self, args):
        self.args_ = args
        self.utc_time_millis_a_ = 0
        self.last_utc_time_millis_a_ = 0
        self.utc_time_millis_g_ = 0
        self.last_utc_time_millis_g_ = 0
        self.f_ib_b_ = np.zeros((3, 1))
        self.omega_ib_b_ = np.zeros((3, 1))
        self.alpha_ib_b_ = np.zeros((3, 1))  # attitude increment (rad)
        self.C_new_old_ = None
        self.C_b_i_ = None

    @property
    def f_ib_b(self):
        return self.f_ib_b_

    @property
    def omega_ib_b(self):
        return self.omega_ib_b_

    @f_ib_b.setter
    def f_ib_b(self, val):
        self.f_ib_b_ = val.to_numpy().reshape((3, 1))

    @omega_ib_b.setter
    def omega_ib_b(self, val):
        self.omega_ib_b_ = val.to_numpy().reshape((3, 1))

    def get_attitude_update_matrix(self):
        t_interval = (self.utc_time_millis_g_ - self.last_utc_time_millis_g_) * MS_TO_SEC

        # Calculate attitude increment, magnitude, and skew-symmetric matrix
        alpha_ib_b = self.omega_ib_b_ * t_interval
        mag_alpha = np.linalg.norm(alpha_ib_b)
        Alpha_ib_b = skew_symmetric(alpha_ib_b)

        # Obtain coordinate transformation matrix from the new attitude to the old
        # using Rodrigues' formula, Paul Groves. (5.73)
        if mag_alpha > 1e-8:
            first_order = (math.sin(mag_alpha) / mag_alpha) * Alpha_ib_b
            second_order = ((1 - math.cos(mag_alpha)) / mag_alpha ** 2) * np.matmul(Alpha_ib_b, Alpha_ib_b)
            self.C_new_old_ = np.eye(3) + first_order + second_order
        else:
            self.C_new_old_ = np.eye(3) + Alpha_ib_b

    def update_attitude(self):
        raise NotImplementedError('virtual method specification of base class')


class NavInsEci(NavIns):
    def update_attitude(self):
        self.get_attitude_update_matrix()

        # TODO 1: change to quaternion representation
        # TODO 2: add coordinate transformation matrix initialization logic
        self.C_b_i_ = np.matmul(self.C_b_i_, self.C_new_old_)


